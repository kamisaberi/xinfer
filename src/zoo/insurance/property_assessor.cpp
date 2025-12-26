#include <xinfer/zoo/insurance/property_assessor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::insurance {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PropertyAssessor::Impl {
    AssessorConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> seg_engine_, cls_engine_, det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> seg_preproc_, cls_preproc_, det_preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> seg_postproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;

    // Tensors
    core::Tensor seg_in, seg_out;
    core::Tensor cls_in, cls_out;
    core::Tensor det_in, det_out;

    // Labels
    std::vector<std::string> cls_labels_, det_labels_;

    Impl(const AssessorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // --- Init Roof Segmenter ---
        seg_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config seg_cfg; seg_cfg.model_path = config_.roof_seg_model_path;
        if (!seg_engine_->load_model(seg_cfg.model_path)) throw std::runtime_error("Failed to load Seg model.");

        seg_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig sp_cfg;
        sp_cfg.target_width = config_.seg_input_size;
        sp_cfg.target_height = config_.seg_input_size;
        seg_preproc_->init(sp_cfg);

        seg_postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig spost_cfg;
        spost_cfg.target_width = config_.seg_input_size;
        spost_cfg.target_height = config_.seg_input_size;
        seg_postproc_->init(spost_cfg);

        // --- Init Roof Classifier ---
        cls_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config c_cfg; c_cfg.model_path = config_.roof_cls_model_path;
        if (!cls_engine_->load_model(c_cfg.model_path)) throw std::runtime_error("Failed to load Cls model.");

        cls_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig cp_cfg;
        cp_cfg.target_width = config_.cls_input_size;
        cp_cfg.target_height = config_.cls_input_size;
        cls_preproc_->init(cp_cfg);

        cls_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cpost_cfg;
        cpost_cfg.top_k = 1;
        load_labels(config_.roof_cls_labels_path, cls_labels_);
        cpost_cfg.labels = cls_labels_;
        cls_postproc_->init(cpost_cfg);

        // --- Init Attribute Detector ---
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.attr_det_model_path;
        if (!det_engine_->load_model(d_cfg.model_path)) throw std::runtime_error("Failed to load Det model.");

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = 640; // Standard YOLO
        dp_cfg.target_height = 640;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        load_labels(config_.attr_det_labels_path, det_labels_);
        dpost_cfg.num_classes = det_labels_.size();
        det_postproc_->init(dpost_cfg);
    }

    void load_labels(const std::string& path, std::vector<std::string>& list) {
        if (path.empty()) return;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) list.push_back(line);
    }
};

// =================================================================================
// Public API
// =================================================================================

PropertyAssessor::PropertyAssessor(const AssessorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PropertyAssessor::~PropertyAssessor() = default;
PropertyAssessor::PropertyAssessor(PropertyAssessor&&) noexcept = default;
PropertyAssessor& PropertyAssessor::operator=(PropertyAssessor&&) noexcept = default;

AssessmentResult PropertyAssessor::assess(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("PropertyAssessor is null.");

    AssessmentResult result;
    result.annotated_image = image.clone();

    preproc::ImageFrame frame{image.data, image.cols, image.rows, preproc::ImageFormat::BGR};

    // --- 1. Roof Segmentation & Condition ---
    pimpl_->seg_preproc_->process(frame, pimpl_->seg_in);
    pimpl_->seg_engine_->predict({pimpl_->seg_in}, {pimpl_->seg_out});
    auto seg_res = pimpl_->seg_postproc_->process(pimpl_->seg_out);

    // Get mask
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));
    cv::Mat roof_mask;
    cv::resize(mask_low, roof_mask, image.size(), 0, 0, cv::INTER_NEAREST);

    // Calculate area
    float roof_pixels = cv::countNonZero(roof_mask);
    result.roof_area_sq_meters = roof_pixels * pimpl_->config_.sq_meters_per_pixel * pimpl_->config_.sq_meters_per_pixel;

    // Crop roof for classifier
    if (roof_pixels > 100) {
        cv::Mat roof_only;
        image.copyTo(roof_only, roof_mask);

        // Classify Condition
        preproc::ImageFrame crop_frame{roof_only.data, roof_only.cols, roof_only.rows, preproc::ImageFormat::BGR};
        pimpl_->cls_preproc_->process(crop_frame, pimpl_->cls_in);
        pimpl_->cls_engine_->predict({pimpl_->cls_in}, {pimpl_->cls_out});
        auto cls_res = pimpl_->cls_postproc_->process(pimpl_->cls_out);

        if (!cls_res.empty() && !cls_res[0].empty()) {
            result.roof_condition = cls_res[0][0].label;
            result.roof_condition_confidence = cls_res[0][0].score;
        }
    }

    // --- 2. Attribute Detection ---
    pimpl_->det_preproc_->process(frame, pimpl_->det_in);
    pimpl_->det_engine_->predict({pimpl_->det_in}, {pimpl_->det_out});
    auto detections = pimpl_->det_postproc_->process({pimpl_->det_out});

    // Scale & Check
    float sx = (float)image.cols / pimpl_->config_.det_preproc_->get_config().target_width; // Assuming a getter for config
    float sy = (float)image.rows / pimpl_->config_.det_preproc_->get_config().target_height;

    for (const auto& det : detections) {
        if (det.class_id < pimpl_->det_labels_.size()) {
            std::string label = pimpl_->det_labels_[det.class_id];
            if (label == "pool") result.attributes.has_swimming_pool = true;
            if (label == "solar") result.attributes.has_solar_panels = true;
            if (label == "tree") result.attributes.has_overhanging_trees = true;
        }
    }

    // --- 3. Visualization ---
    // (Draw roof mask and detected attribute boxes)
    cv::Mat color_mask = cv::Mat(image.size(), CV_8UC3, cv::Scalar(0, 0, 128)); // Red for roof
    cv::addWeighted(result.annotated_image, 1.0, color_mask, 0.3, 0.0, result.annotated_image, roof_mask);

    return result;
}

} // namespace xinfer::zoo::insurance