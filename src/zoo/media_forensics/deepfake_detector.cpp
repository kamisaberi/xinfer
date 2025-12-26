#include <xinfer/zoo/media_forensics/deepfake_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::media_forensics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DeepfakeDetector::Impl {
    DeepfakeConfig config_;

    // --- Pipeline 1: Face Detection ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    core::Tensor det_input, det_output;

    // --- Pipeline 2: Deepfake Classification ---
    std::unique_ptr<backends::IBackend> cls_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> cls_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;
    core::Tensor cls_input, cls_output;

    Impl(const DeepfakeConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg;
        d_cfg.model_path = config_.detector_model_path;
        d_cfg.vendor_params = config_.vendor_params; // Pass params (like DeviceID)

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("DeepfakeDetector: Failed to load detector " + d_cfg.model_path);
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.det_input_size;
        dp_cfg.target_height = config_.det_input_size;
        dp_cfg.target_format = preproc::ImageFormat::RGB;
        dp_cfg.layout_nchw = true;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = 0.5f;
        dpost_cfg.nms_threshold = 0.45f;
        dpost_cfg.num_classes = 1; // Face
        det_postproc_->init(dpost_cfg);

        // 2. Setup Classifier
        cls_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config c_cfg;
        c_cfg.model_path = config_.classifier_model_path;
        // Reuse vendor params (e.g. same GPU)
        c_cfg.vendor_params = config_.vendor_params;

        if (!cls_engine_->load_model(c_cfg.model_path)) {
            throw std::runtime_error("DeepfakeDetector: Failed to load classifier " + c_cfg.model_path);
        }

        cls_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig cp_cfg;
        cp_cfg.target_width = config_.cls_input_size;
        cp_cfg.target_height = config_.cls_input_size;
        cp_cfg.target_format = preproc::ImageFormat::RGB;
        cp_cfg.layout_nchw = true;

        // Deepfake models are sensitive to noise, use standard ImageNet normalization
        cp_cfg.norm_params.mean = config_.mean;
        cp_cfg.norm_params.std = config_.std;
        cp_cfg.norm_params.scale_factor = 1.0f; // Assumes inputs 0-255

        cls_preproc_->init(cp_cfg);

        cls_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cpost_cfg;
        cpost_cfg.top_k = 2; // Real/Fake
        cpost_cfg.apply_softmax = true;
        // Assume model outputs [Real, Fake]
        // This mapping depends heavily on training.
        // Index 0: Real, Index 1: Fake is common binary classification
        cpost_cfg.labels = {"Real", "Fake"};
        cls_postproc_->init(cpost_cfg);
    }

    FaceAnalysis classify_face(const cv::Mat& face_crop, const postproc::BoundingBox& box) {
        FaceAnalysis result;
        result.box = box;

        // Preprocess Crop
        preproc::ImageFrame crop_frame;
        crop_frame.data = face_crop.data;
        crop_frame.width = face_crop.cols;
        crop_frame.height = face_crop.rows;
        crop_frame.format = preproc::ImageFormat::BGR; // OpenCV crop is BGR

        cls_preproc_->process(crop_frame, cls_input);

        // Inference
        cls_engine_->predict({cls_input}, {cls_output});

        // Postprocess
        auto batch_res = cls_postproc_->process(cls_output);

        // Parse Results
        // We look for the "Fake" class score
        float score_fake = 0.0f;
        float score_real = 0.0f;

        if (!batch_res.empty() && !batch_res[0].empty()) {
            for (const auto& r : batch_res[0]) {
                if (r.label == "Fake" || r.id == 1) score_fake = r.score;
                if (r.label == "Real" || r.id == 0) score_real = r.score;
            }
        }

        // If single output node (sigmoid)
        if (cls_output.size() == 1) {
            float* ptr = (float*)cls_output.data();
            score_fake = ptr[0];
            score_real = 1.0f - score_fake;
        }

        result.fake_score = score_fake;
        result.is_fake = (score_fake > config_.fake_threshold);
        result.label = result.is_fake ? "Fake" : "Real";

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DeepfakeDetector::DeepfakeDetector(const DeepfakeConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DeepfakeDetector::~DeepfakeDetector() = default;
DeepfakeDetector::DeepfakeDetector(DeepfakeDetector&&) noexcept = default;
DeepfakeDetector& DeepfakeDetector::operator=(DeepfakeDetector&&) noexcept = default;

std::vector<FaceAnalysis> DeepfakeDetector::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DeepfakeDetector is null.");

    std::vector<FaceAnalysis> results;

    // 1. Detect Faces
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});
    auto detections = pimpl_->det_postproc_->process({pimpl_->det_output});

    // 2. Classify Each Face
    float scale_x = (float)image.cols / pimpl_->config_.det_input_size;
    float scale_y = (float)image.rows / pimpl_->config_.det_input_size;

    for (const auto& det : detections) {
        // Scale Box
        int x1 = std::max(0, (int)(det.x1 * scale_x));
        int y1 = std::max(0, (int)(det.y1 * scale_y));
        int x2 = std::min(image.cols, (int)(det.x2 * scale_x));
        int y2 = std::min(image.rows, (int)(det.y2 * scale_y));

        // Skip tiny faces
        if (x2 - x1 < 32 || y2 - y1 < 32) continue;

        // Crop with margin?
        // Deepfake artifacts often appear on boundaries, so a tight crop is usually fine,
        // but some models prefer 1.2x context. Keeping tight for now.
        cv::Rect roi(x1, y1, x2 - x1, y2 - y1);
        cv::Mat face_crop = image(roi);

        // Store original box coords in result
        postproc::BoundingBox original_box = det;
        original_box.x1 = (float)x1; original_box.y1 = (float)y1;
        original_box.x2 = (float)x2; original_box.y2 = (float)y2;

        results.push_back(pimpl_->classify_face(face_crop, original_box));
    }

    return results;
}

} // namespace xinfer::zoo::media_forensics