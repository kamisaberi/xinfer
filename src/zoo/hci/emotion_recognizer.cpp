#include <xinfer/zoo/hci/emotion_recognizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

// We reuse the generic vision detector
#include <xinfer/zoo/vision/face_detector.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::hci {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct EmotionRecognizer::Impl {
    EmotionConfig config_;

    // --- Components ---
    std::unique_ptr<vision::FaceDetector> face_detector_;
    std::unique_ptr<backends::IBackend> cls_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> cls_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;

    // --- Tensors ---
    core::Tensor cls_input, cls_output;

    Impl(const EmotionConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Face Detector (if provided)
        if (!config_.detector_path.empty()) {
            vision::FaceDetectorConfig det_cfg;
            det_cfg.target = config_.target;
            det_cfg.model_path = config_.detector_path;
            face_detector_ = std::make_unique<vision::FaceDetector>(det_cfg);
        }

        // 2. Setup Emotion Classifier
        cls_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config c_cfg; c_cfg.model_path = config_.classifier_path;

        if (!cls_engine_->load_model(c_cfg.model_path)) {
            throw std::runtime_error("EmotionRecognizer: Failed to load classifier.");
        }

        cls_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig cp_cfg;
        cp_cfg.target_width = config_.input_width;
        cp_cfg.target_height = config_.input_height;
        cp_cfg.target_format = preproc::ImageFormat::GRAY; // Emotion models are often grayscale
        cp_cfg.layout_nchw = true;
        // Simple 0-1 scaling for grayscale
        cp_cfg.norm_params.scale_factor = 1.0f / 255.0f;
        cls_preproc_->init(cp_cfg);

        cls_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cpost_cfg;
        cpost_cfg.top_k = config_.labels.size(); // Get all scores
        cpost_cfg.apply_softmax = true;
        cpost_cfg.labels = config_.labels;
        cls_postproc_->init(cpost_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

EmotionRecognizer::EmotionRecognizer(const EmotionConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

EmotionRecognizer::~EmotionRecognizer() = default;
EmotionRecognizer::EmotionRecognizer(EmotionRecognizer&&) noexcept = default;
EmotionRecognizer& EmotionRecognizer::operator=(EmotionRecognizer&&) noexcept = default;

std::vector<std::pair<cv::Rect, EmotionResult>> EmotionRecognizer::recognize(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("EmotionRecognizer is null.");

    std::vector<cv::Rect> face_boxes;

    // 1. Find Faces
    if (pimpl_->face_detector_) {
        auto detections = pimpl_->face_detector_->detect(image);
        for (const auto& det : detections) {
            face_boxes.push_back(det.to_rect());
        }
    } else {
        // If no detector, assume the whole image is the face
        face_boxes.push_back(cv::Rect(0, 0, image.cols, image.rows));
    }

    if (face_boxes.empty()) return {};

    // 2. Classify Each Face
    std::vector<std::pair<cv::Rect, EmotionResult>> all_results;

    for (const auto& box : face_boxes) {
        // Crop face
        cv::Rect roi = box & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat face_crop = image(roi);

        // Preprocess
        preproc::ImageFrame frame;
        frame.data = face_crop.data;
        frame.width = face_crop.cols;
        frame.height = face_crop.rows;
        frame.format = (face_crop.channels() == 1) ? preproc::ImageFormat::GRAY : preproc::ImageFormat::BGR;

        pimpl_->cls_preproc_->process(frame, pimpl_->cls_input);

        // Inference
        pimpl_->cls_engine_->predict({pimpl_->cls_input}, {pimpl_->cls_output});

        // Postprocess
        auto results = pimpl_->cls_postproc_->process(pimpl_->cls_output);

        EmotionResult res;
        if (!results.empty() && !results[0].empty()) {
            // Top result
            res.dominant_emotion = results[0][0].label;
            res.confidence = results[0][0].score;

            // Fill scores map
            for (const auto& r : results[0]) {
                res.scores[r.label] = r.score;
            }
        }

        all_results.push_back({box, res});
    }

    return all_results;
}

} // namespace xinfer::zoo::hci