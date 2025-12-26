#include <xinfer/zoo/vision/pose_estimator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; generic object detector doesn't handle keypoints logic.
// We implement custom decoding here.

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PoseEstimator::Impl {
    PoseEstimatorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const PoseEstimatorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PoseEstimator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);
    }

    // --- Custom Decoding for YOLO-Pose ---
    std::vector<PoseResult> decode(const core::Tensor& tensor, const cv::Size& original_size) {
        std::vector<PoseResult> results;

        // Output Shape: [1, Channels, Anchors]
        // Channels = 4 (Box) + 1 (Score) + (NumKpts * 3)
        // e.g., 17 kpts -> 4 + 1 + 51 = 56 channels.

        auto shape = tensor.shape();
        int num_channels = (int)shape[1];
        int num_anchors  = (int)shape[2];
        int num_kpts     = config_.num_keypoints;

        const float* data = static_cast<const float*>(tensor.data());

        // Pointers to strided data (Column-Major per anchor)
        const float* p_cx = data;
        const float* p_cy = data + num_anchors;
        const float* p_w  = data + num_anchors * 2;
        const float* p_h  = data + num_anchors * 3;
        const float* p_score = data + num_anchors * 4;
        const float* p_kpts  = data + num_anchors * 5; // Start of keypoints

        // Intermediate lists for NMS
        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<std::vector<Keypoint>> all_kpts;

        for (int i = 0; i < num_anchors; ++i) {
            float score = p_score[i];
            if (score < config_.conf_threshold) continue;

            // Decode Box
            float cx = p_cx[i];
            float cy = p_cy[i];
            float w  = p_w[i];
            float h  = p_h[i];

            int x = (int)(cx - w * 0.5f);
            int y = (int)(cy - h * 0.5f);
            int width = (int)w;
            int height = (int)h;

            boxes.push_back(cv::Rect(x, y, width, height));
            scores.push_back(score);

            // Decode Keypoints
            std::vector<Keypoint> kpts;
            kpts.reserve(num_kpts);
            for (int k = 0; k < num_kpts; ++k) {
                // Each keypoint has 3 values: x, y, conf
                // Stride is num_anchors
                float kx = p_kpts[(k * 3 + 0) * num_anchors + i];
                float ky = p_kpts[(k * 3 + 1) * num_anchors + i];
                float kc = p_kpts[(k * 3 + 2) * num_anchors + i];
                kpts.push_back({kx, ky, kc});
            }
            all_kpts.push_back(std::move(kpts));
        }

        // Apply NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, scores, config_.conf_threshold, config_.nms_threshold, indices);

        // Scale Factor
        float scale_x = (float)original_size.width / config_.input_width;
        float scale_y = (float)original_size.height / config_.input_height;

        for (int idx : indices) {
            PoseResult res;

            // Box Scaling
            res.box.x1 = boxes[idx].x * scale_x;
            res.box.y1 = boxes[idx].y * scale_y;
            res.box.x2 = (boxes[idx].x + boxes[idx].width) * scale_x;
            res.box.y2 = (boxes[idx].y + boxes[idx].height) * scale_y;
            res.box.confidence = scores[idx];
            res.box.class_id = 0; // Person

            // Keypoint Scaling
            for (const auto& kp : all_kpts[idx]) {
                Keypoint scaled_kp;
                scaled_kp.x = kp.x * scale_x;
                scaled_kp.y = kp.y * scale_y;
                scaled_kp.confidence = kp.confidence;
                res.keypoints.push_back(scaled_kp);
            }

            results.push_back(res);
        }

        return results;
    }
};

// =================================================================================
// Public API
// =================================================================================

PoseEstimator::PoseEstimator(const PoseEstimatorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PoseEstimator::~PoseEstimator() = default;
PoseEstimator::PoseEstimator(PoseEstimator&&) noexcept = default;
PoseEstimator& PoseEstimator::operator=(PoseEstimator&&) noexcept = default;

std::vector<PoseResult> PoseEstimator::estimate(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("PoseEstimator is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Custom Decode)
    return pimpl_->decode(pimpl_->output_tensor, image.size());
}

void PoseEstimator::draw_skeleton(cv::Mat& image, const std::vector<PoseResult>& results) {
    // Standard COCO Skeleton Connections
    static const std::vector<std::pair<int, int>> skeleton = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, // Limbs
        {11, 12}, {5, 11}, {6, 12}, // Body
        {5, 6}, {5, 7}, {6, 8}, {7, 9}, {8, 10}, // Arms
        {1, 2}, {0, 1}, {0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6} // Head
    };

    for (const auto& res : results) {
        // Draw Box
        cv::rectangle(image, cv::Point(res.box.x1, res.box.y1), cv::Point(res.box.x2, res.box.y2), cv::Scalar(0, 255, 0), 2);

        // Draw Keypoints
        for (const auto& kp : res.keypoints) {
            if (kp.confidence > 0.5f) {
                cv::circle(image, cv::Point(kp.x, kp.y), 3, cv::Scalar(0, 0, 255), -1);
            }
        }

        // Draw Limbs
        for (const auto& pair : skeleton) {
            if (pair.first < res.keypoints.size() && pair.second < res.keypoints.size()) {
                const auto& kp1 = res.keypoints[pair.first];
                const auto& kp2 = res.keypoints[pair.second];

                if (kp1.confidence > 0.5f && kp2.confidence > 0.5f) {
                    cv::line(image, cv::Point(kp1.x, kp1.y), cv::Point(kp2.x, kp2.y), cv::Scalar(255, 0, 0), 2);
                }
            }
        }
    }
}

} // namespace xinfer::zoo::vision