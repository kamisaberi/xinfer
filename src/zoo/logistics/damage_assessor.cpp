#include <xinfer/zoo/logistics/damage_assessor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::logistics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DamageAssessor::Impl {
    AssessorConfig config_;

    // --- Detector Components ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    core::Tensor det_input, det_output;
    std::vector<std::string> det_labels_;

    // --- Classifier Components (Optional) ---
    std::unique_ptr<backends::IBackend> cls_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> cls_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;
    core::Tensor cls_input, cls_output;
    std::vector<std::string> cls_labels_;

    Impl(const AssessorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.detector_model_path;
        d_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("DamageAssessor: Failed to load detector " + config_.detector_model_path);
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.input_width;
        dp_cfg.target_height = config_.input_height;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = config_.conf_threshold;
        dpost_cfg.nms_threshold = config_.nms_threshold;
        load_labels(config_.labels_path, det_labels_);
        dpost_cfg.num_classes = det_labels_.size();
        det_postproc_->init(dpost_cfg);

        // 2. Setup Classifier (Optional)
        if (!config_.classifier_model_path.empty()) {
            cls_engine_ = backends::BackendFactory::create(config_.target);
            xinfer::Config c_cfg; c_cfg.model_path = config_.classifier_model_path;

            if (cls_engine_->load_model(c_cfg.model_path)) {
                cls_preproc_ = preproc::create_image_preprocessor(config_.target);
                preproc::ImagePreprocConfig cp_cfg;
                // Classifier might use different size
                cp_cfg.target_width = 224;
                cp_cfg.target_height = 224;
                cls_preproc_->init(cp_cfg);

                cls_postproc_ = postproc::create_classification(config_.target);
                postproc::ClassificationConfig cpost_cfg;
                cpost_cfg.top_k = 1;
                load_labels(config_.classifier_labels_path, cls_labels_);
                cpost_cfg.labels = cls_labels_;
                cls_postproc_->init(cpost_cfg);
            }
        }
    }

    void load_labels(const std::string& path, std::vector<std::string>& list) {
        if (path.empty()) return;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            list.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

DamageAssessor::DamageAssessor(const AssessorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DamageAssessor::~DamageAssessor() = default;
DamageAssessor::DamageAssessor(DamageAssessor&&) noexcept = default;
DamageAssessor& DamageAssessor::operator=(DamageAssessor&&) noexcept = default;

AssessmentResult DamageAssessor::assess(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DamageAssessor is null.");

    AssessmentResult result;
    result.visualization = image.clone();

    // --- 1. Detection of Specific Damages ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});
    auto detections = pimpl_->det_postproc_->process({pimpl_->det_output});

    // Scale & Format
    float sx = (float)image.cols / pimpl_->config_.input_width;
    float sy = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : detections) {
        DamageLocation loc;
        loc.box = det;
        loc.box.x1 *= sx; loc.box.x2 *= sx;
        loc.box.y1 *= sy; loc.box.y2 *= sy;
        loc.confidence = det.confidence;

        if (det.class_id < (int)pimpl_->det_labels_.size()) {
            loc.damage_type = pimpl_->det_labels_[det.class_id];
        } else {
            loc.damage_type = "Unknown Damage";
        }
        result.damages.push_back(loc);

        // Draw on visual
        cv::Rect r((int)loc.box.x1, (int)loc.box.y1, (int)(loc.box.x2-loc.box.x1), (int)(loc.box.y2-loc.box.y1));
        cv::rectangle(result.visualization, r, cv::Scalar(0, 0, 255), 2);
        cv::putText(result.visualization, loc.damage_type, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    }

    // --- 2. Overall Condition (Classifier or Heuristic) ---
    if (pimpl_->cls_engine_) {
        // Run dedicated classifier
        pimpl_->cls_preproc_->process(frame, pimpl_->cls_input);
        pimpl_->cls_engine_->predict({pimpl_->cls_input}, {pimpl_->cls_output});
        auto cls_res = pimpl_->cls_postproc_->process(pimpl_->cls_output);

        if (!cls_res.empty() && !cls_res[0].empty()) {
            result.overall_condition = cls_res[0][0].label;
            result.overall_confidence = cls_res[0][0].score;
        }
    } else {
        // Heuristic: If any damage detected, it's damaged.
        if (!result.damages.empty()) {
            // Count "severe" damages (e.g. Crack > Scratch)
            int severe_count = 0;
            for(const auto& d : result.damages) {
                if (d.damage_type == "Crack" || d.damage_type == "Shatter") severe_count++;
            }

            if (severe_count > 0) {
                result.overall_condition = "Severe Damage";
                result.overall_confidence = 0.9f;
            } else {
                result.overall_condition = "Minor Damage";
                result.overall_confidence = 0.8f;
            }
        } else {
            result.overall_condition = "Undamaged";
            result.overall_confidence = 1.0f;
        }
    }

    return result;
}

} // namespace xinfer::zoo::logistics