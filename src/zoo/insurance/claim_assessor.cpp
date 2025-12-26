#include <xinfer/zoo/insurance/claim_assessor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <fstream>
#include <iostream>
#include <algorithm>

namespace xinfer::zoo::insurance {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ClaimAssessor::Impl {
    AssessorConfig config_;

    // --- Detector Components ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    core::Tensor det_input, det_output;
    std::vector<std::string> det_labels_;

    // --- Peril Classifier Components ---
    std::unique_ptr<backends::IBackend> peril_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> peril_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> peril_postproc_;
    core::Tensor peril_input, peril_output;
    std::vector<std::string> peril_labels_;

    Impl(const AssessorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.detector_model_path;
        d_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("ClaimAssessor: Failed to load detector " + config_.detector_model_path);
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.det_input_width;
        dp_cfg.target_height = config_.det_input_height;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = config_.det_conf_thresh;
        load_labels(config_.damage_labels_path, det_labels_);
        dpost_cfg.num_classes = det_labels_.size();
        det_postproc_->init(dpost_cfg);

        // 2. Setup Peril Classifier
        peril_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config p_cfg; p_cfg.model_path = config_.peril_model_path;

        if (!peril_engine_->load_model(p_cfg.model_path)) {
            throw std::runtime_error("ClaimAssessor: Failed to load peril model " + config_.peril_model_path);
        }

        peril_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pp_cfg;
        pp_cfg.target_width = config_.peril_input_width;
        pp_cfg.target_height = config_.peril_input_height;
        peril_preproc_->init(pp_cfg);

        peril_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig ppost_cfg;
        ppost_cfg.top_k = 1;
        load_labels(config_.peril_labels_path, peril_labels_);
        ppost_cfg.labels = peril_labels_;
        peril_postproc_->init(ppost_cfg);
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

ClaimAssessor::ClaimAssessor(const AssessorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ClaimAssessor::~ClaimAssessor() = default;
ClaimAssessor::ClaimAssessor(ClaimAssessor&&) noexcept = default;
ClaimAssessor& ClaimAssessor::operator=(ClaimAssessor&&) noexcept = default;

ClaimAssessment ClaimAssessor::assess(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ClaimAssessor is null.");

    ClaimAssessment result;
    result.visualization = image.clone();

    // --- 1. Classify Primary Peril ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->peril_preproc_->process(frame, pimpl_->peril_input);
    pimpl_->peril_engine_->predict({pimpl_->peril_input}, {pimpl_->peril_output});
    auto peril_res = pimpl_->peril_postproc_->process(pimpl_->peril_output);

    if (!peril_res.empty() && !peril_res[0].empty()) {
        result.primary_peril = peril_res[0][0].label;
        result.peril_confidence = peril_res[0][0].score;
    }

    // --- 2. Detect Specific Damages ---
    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});
    auto detections = pimpl_->det_postproc_->process({pimpl_->det_output});

    // Scale & Format
    float sx = (float)image.cols / pimpl_->config_.det_input_width;
    float sy = (float)image.rows / pimpl_->config_.det_input_height;

    int severe_damage_count = 0;
    for (const auto& det : detections) {
        DamageLocation loc;
        loc.box = det;
        loc.box.x1 *= sx; loc.box.x2 *= sx;
        loc.box.y1 *= sy; loc.box.y2 *= sy;
        loc.confidence = det.confidence;

        if (det.class_id < (int)pimpl_->det_labels_.size()) {
            loc.damage_type = pimpl_->det_labels_[det.class_id];
        }

        // Severity Heuristic
        // "Totaled" car, "Broken Glass", etc. are high severity
        if (loc.damage_type.find("total") != std::string::npos ||
            loc.damage_type.find("broken") != std::string::npos ||
            loc.damage_type.find("shatter") != std::string::npos) {
            severe_damage_count++;
        }

        result.damages.push_back(loc);
    }

    // --- 3. Determine Overall Severity ---
    if (result.primary_peril == "Fire" || result.primary_peril == "Flood") {
        result.severity = SeverityLevel::TOTAL_LOSS;
    }
    else if (severe_damage_count > 0 || result.damages.size() > 5) {
        result.severity = SeverityLevel::HIGH;
    }
    else if (!result.damages.empty()) {
        result.severity = SeverityLevel::MEDIUM;
    } else {
        result.severity = SeverityLevel::LOW; // No detected damage, but might be other issues
    }

    if(result.primary_peril == "Undamaged") result.severity = SeverityLevel::LOW;

    // --- 4. Visualization ---
    for (const auto& loc : result.damages) {
        cv::Rect r((int)loc.box.x1, (int)loc.box.y1, (int)(loc.box.x2-loc.box.x1), (int)(loc.box.y2-loc.box.y1));
        cv::rectangle(result.visualization, r, cv::Scalar(0, 0, 255), 2);
        cv::putText(result.visualization, loc.damage_type, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0,0,255), 2);
    }

    std::string summary = "Peril: " + result.primary_peril;
    cv::putText(result.visualization, summary, cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(255,255,255), 2);

    return result;
}

} // namespace xinfer::zoo::insurance