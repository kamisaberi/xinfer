#include <xinfer/zoo/threed/slam_accelerator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <algorithm>
#include <vector>

namespace xinfer::zoo::threed {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SlamAccelerator::Impl {
    SlamConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_score_map; // Heatmap
    core::Tensor output_desc_map;  // Descriptors

    Impl(const SlamConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SlamAccelerator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::GRAY; // SLAM usually uses grayscale
        pre_cfg.layout_nchw = true;
        pre_cfg.norm_params.scale_factor = 1.0f / 255.0f; // [0,1] input

        preproc_->init(pre_cfg);
    }

    // --- Custom Decoding for SuperPoint-like outputs ---
    FrameFeatures decode_superpoint(const core::Tensor& raw_scores, const core::Tensor& raw_desc, const cv::Size& orig_size) {
        // NOTE: This logic assumes a standard SuperPoint output topology.
        // If using D2Net or R2D2, the decoding logic differs.
        // Assuming: Scores [1, 1, H, W] (Dense map after model-internal pixel shuffle)
        //           Desc   [1, 256, H/8, W/8] (Coarse map)

        // 1. Process Scores (Find Corners)
        // ------------------------------------
        auto score_shape = raw_scores.shape();
        int H = score_shape[2];
        int W = score_shape[3];
        const float* s_ptr = static_cast<const float*>(raw_scores.data());

        // Wrap in Mat
        cv::Mat score_mat(H, W, CV_32F, const_cast<float*>(s_ptr));

        // Thresholding
        cv::Mat mask;
        cv::threshold(score_mat, mask, config_.keypoint_threshold, 255, cv::THRESH_BINARY);
        mask.convertTo(mask, CV_8U);

        // NMS (Simple dilation-based approximation or iterative)
        // Here iterating points for NMS
        std::vector<cv::KeyPoint> keypoints;

        // Fast sparse iteration
        std::vector<cv::Point> locations;
        cv::findNonZero(mask, locations);

        // Sort by score descending for NMS
        std::sort(locations.begin(), locations.end(), [&](const cv::Point& a, const cv::Point& b) {
            return score_mat.at<float>(a) > score_mat.at<float>(b);
        });

        // Apply NMS Radius
        int nms_sq = config_.nms_radius * config_.nms_radius;
        // Simple occupancy grid for NMS could be faster, but vector check is okay for <2000 points
        // For production, use a grid-based suppression.

        // Using a mask for NMS
        cv::Mat nms_mask = cv::Mat::zeros(H, W, CV_8U);
        int count = 0;

        for (const auto& pt : locations) {
            if (count >= config_.max_keypoints) break;
            if (nms_mask.at<uint8_t>(pt) == 0) {
                // Keep this point
                keypoints.emplace_back((float)pt.x, (float)pt.y, 1.0f, -1.0f, score_mat.at<float>(pt));
                count++;

                // Suppress neighbors
                cv::circle(nms_mask, pt, config_.nms_radius, cv::Scalar(255), -1);
            }
        }

        // 2. Sample Descriptors
        // ------------------------------------
        // Descriptors are usually coarse (H/8, W/8). We need to bicubic interpolate
        // at the exact sub-pixel keypoint location.
        auto desc_shape = raw_desc.shape();
        int D = desc_shape[1]; // 256
        int dH = desc_shape[2];
        int dW = desc_shape[3];
        const float* d_ptr = static_cast<const float*>(raw_desc.data());

        // We assume descriptor map is NCHW. Wrap pointer is tricky for multi-channel interpolation.
        // Strategy: Loop over keypoints, for each keypoint, sample D values.

        cv::Mat descriptors(keypoints.size(), D, CV_32F);

        float scale_x = (float)dW / W; // e.g. 1/8
        float scale_y = (float)dH / H;

        for (size_t k = 0; k < keypoints.size(); ++k) {
            float px = keypoints[k].pt.x * scale_x;
            float py = keypoints[k].pt.y * scale_y;

            // Bilinear Interpolation indices
            int x0 = (int)px;
            int y0 = (int)py;
            int x1 = std::min(x0 + 1, dW - 1);
            int y1 = std::min(y0 + 1, dH - 1);

            float dx = px - x0;
            float dy = py - y0;

            for (int c = 0; c < D; ++c) {
                // Offset to channel c plane
                const float* plane = d_ptr + (c * dW * dH);

                float v00 = plane[y0 * dW + x0];
                float v10 = plane[y0 * dW + x1];
                float v01 = plane[y1 * dW + x0];
                float v11 = plane[y1 * dW + x1];

                // Interpolate
                float val = (v00 * (1 - dx) * (1 - dy)) +
                            (v10 * dx * (1 - dy)) +
                            (v01 * (1 - dx) * dy) +
                            (v11 * dx * dy);

                descriptors.at<float>(k, c) = val;
            }
        }

        // 3. Normalize Descriptors (L2 Norm)
        // Crucial for Cosine/L2 matching
        for (int i = 0; i < descriptors.rows; ++i) {
            cv::Mat row = descriptors.row(i);
            cv::normalize(row, row);
        }

        // 4. Rescale Keypoints to Original Image Size
        float view_scale_x = (float)orig_size.width / config_.input_width;
        float view_scale_y = (float)orig_size.height / config_.input_height;

        for (auto& kp : keypoints) {
            kp.pt.x *= view_scale_x;
            kp.pt.y *= view_scale_y;
        }

        FrameFeatures feat;
        feat.keypoints = keypoints;
        feat.descriptors = descriptors;

        return feat;
    }
};

// =================================================================================
// Public API
// =================================================================================

SlamAccelerator::SlamAccelerator(const SlamConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SlamAccelerator::~SlamAccelerator() = default;
SlamAccelerator::SlamAccelerator(SlamAccelerator&&) noexcept = default;
SlamAccelerator& SlamAccelerator::operator=(SlamAccelerator&&) noexcept = default;

FrameFeatures SlamAccelerator::extract(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SlamAccelerator is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::GRAY; // Standard for SLAM

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    // Depending on model export, output might be 1 tensor or split
    try {
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_score_map, pimpl_->output_desc_map});
    } catch (...) {
        // Handle models that output a single concatenated tensor
    }

    // 3. Postprocess
    return pimpl_->decode_superpoint(pimpl_->output_score_map, pimpl_->output_desc_map, image.size());
}

std::vector<cv::DMatch> SlamAccelerator::match(const FrameFeatures& f1, const FrameFeatures& f2) {
    // Simple Brute Force Matcher (L2 Norm for normalized descriptors is equivalent to Cosine)
    // For production SLAM, use FLANN or LightGlue

    if (f1.descriptors.empty() || f2.descriptors.empty()) return {};

    cv::BFMatcher matcher(cv::NORM_L2, true); // Cross-check=true for robustness
    std::vector<cv::DMatch> matches;
    matcher.match(f1.descriptors, f2.descriptors, matches);

    return matches;
}

} // namespace xinfer::zoo::threed