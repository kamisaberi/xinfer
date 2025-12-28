#include <xinfer/zoo/accessibility/sign_translator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We compose other Zoo modules ---
#include <xinfer/zoo/vision/pose_estimator.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <deque>
#include <numeric>

namespace xinfer::zoo::accessibility {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SignTranslator::Impl {
    SignConfig config_;

    // --- Components ---
    std::unique_ptr<vision::PoseEstimator> pose_estimator_;
    std::unique_ptr<backends::IBackend> classifier_engine_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // --- Data ---
    // Buffer for a sequence of keypoints
    // Flattened: [Frame0_Kpt0_x, Frame0_Kpt0_y, ..., Frame29_KptN_y]
    std::deque<std::vector<float>> sequence_buffer_;

    core::Tensor input_tensor;
    core::Tensor output_tensor;

    std::vector<std::string> labels_;

    // State
    std::string last_word_ = "";

    Impl(const SignConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init Pose Estimator
        vision::PoseEstimatorConfig pose_cfg;
        pose_cfg.target = config_.target;
        pose_cfg.model_path = config_.pose_model_path;
        pose_estimator_ = std::make_unique<vision::PoseEstimator>(pose_cfg);

        // 2. Init Sign Classifier
        classifier_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config cls_cfg; cls_cfg.model_path = config_.classifier_model_path;
        if (!classifier_engine_->load_model(cls_cfg.model_path)) {
            throw std::runtime_error("SignTranslator: Failed to load classifier model.");
        }

        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 1;
        post_cfg.apply_softmax = true;
        load_labels(config_.labels_path);
        post_cfg.labels = labels_;
        postproc_->init(post_cfg);
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (file.is_open()) {
            std::string line;
            while(std::getline(file, line)) labels_.push_back(line);
        }
    }

    // --- Core Logic ---
    void update_buffer(const std::vector<vision::PoseResult>& poses) {
        std::vector<float> current_frame_kpts;

        if (!poses.empty()) {
            // Use first person's pose
            const auto& pose = poses[0];

            // Flatten keypoints: [x0,y0,c0, x1,y1,c1, ...]
            // Normalize relative to the torso for translation invariance
            const auto& l_shoulder = pose.keypoints[5];
            const auto& r_shoulder = pose.keypoints[6];

            if (l_shoulder.confidence > 0.5 && r_shoulder.confidence > 0.5) {
                float center_x = (l_shoulder.x + r_shoulder.x) / 2.0f;
                float center_y = (l_shoulder.y + r_shoulder.y) / 2.0f;
                float scale = std::abs(l_shoulder.x - r_shoulder.x);
                if (scale < 1e-3) scale = 1.0f;

                for (const auto& kp : pose.keypoints) {
                    current_frame_kpts.push_back((kp.x - center_x) / scale);
                    current_frame_kpts.push_back((kp.y - center_y) / scale);
                    current_frame_kpts.push_back(kp.confidence);
                }
            }
        }

        // Pad if no pose was detected
        if (current_frame_kpts.empty()) {
            // Size = NumKeypoints * 3
            // Assuming 17 keypoints from YOLO-Pose
            current_frame_kpts.assign(17 * 3, 0.0f);
        }

        // Update buffer
        sequence_buffer_.push_back(current_frame_kpts);
        if (sequence_buffer_.size() > (size_t)config_.sequence_length) {
            sequence_buffer_.pop_front();
        }
    }

    void prepare_tensor() {
        // Flatten the deque of vectors into a single tensor
        // Shape: [1, SeqLen, KptDim]
        int kpt_dim = sequence_buffer_.front().size();
        input_tensor.resize({1, (int64_t)config_.sequence_length, (int64_t)kpt_dim}, core::DataType::kFLOAT);

        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        for (const auto& frame_kpts : sequence_buffer_) {
            std::memcpy(ptr + idx, frame_kpts.data(), frame_kpts.size() * sizeof(float));
            idx += kpt_dim;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

SignTranslator::SignTranslator(const SignConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SignTranslator::~SignTranslator() = default;
SignTranslator::SignTranslator(SignTranslator&&) noexcept = default;
SignTranslator& SignTranslator::operator=(SignTranslator&&) noexcept = default;

void SignTranslator::reset() {
    if (pimpl_) {
        pimpl_->sequence_buffer_.clear();
        pimpl_->last_word_ = "";
    }
}

SignResult SignTranslator::translate(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SignTranslator is null.");

    // 1. Get Pose
    auto poses = pimpl_->pose_estimator_->estimate(image);

    // 2. Update Buffer
    pimpl_->update_buffer(poses);

    // 3. Check if ready for inference
    if (pimpl_->sequence_buffer_.size() < (size_t)pimpl_->config_.sequence_length) {
        return {"(Collecting data...)", 0.0f, false};
    }

    // 4. Prepare Input Tensor
    pimpl_->prepare_tensor();

    // 5. Inference
    pimpl_->classifier_engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 6. Postprocess
    auto results = pimpl_->postproc_->process(pimpl_->output_tensor);

    SignResult res;
    if (!results.empty() && !results[0].empty()) {
        const auto& top1 = results[0][0];

        // Avoid spamming the same word
        // In a real app, you'd check for a "blank" or "no-sign" class
        if (top1.label != pimpl_->last_word_ && top1.confidence > pimpl_->config_.min_confidence) {
            res.current_word = top1.label;
            res.confidence = top1.score;
            res.is_new_word = true;
            pimpl_->last_word_ = res.current_word;
        } else {
            res.current_word = pimpl_->last_word_;
            res.confidence = top1.score;
            res.is_new_word = false;
        }
    }

    return res;
}

} // namespace xinfer::zoo::accessibility