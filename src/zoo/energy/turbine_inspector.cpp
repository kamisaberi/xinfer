#include <xinfer/zoo/energy/turbine_inspector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

// We reuse the generic detector
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <fstream>

namespace xinfer::zoo::energy {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TurbineInspector::Impl {
    TurbineConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> seg_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> seg_preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> seg_postproc_;

    // We reuse the high-level Detector Zoo module
    std::unique_ptr<vision::ObjectDetector> detector_;

    // --- Tensors ---
    core::Tensor seg_in, seg_out;

    Impl(const TurbineConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Blade Segmenter
        seg_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config s_cfg; s_cfg.model_path = config_.seg_model_path;

        if (!seg_engine_->load_model(s_cfg.model_path)) {
            throw std::runtime_error("TurbineInspector: Failed to load Seg model.");
        }

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

        // 2. Setup Damage Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.det_model_path;
        det_cfg.labels_path = config_.det_labels_path;
        det_cfg.input_width = config_.det_input_width;
        det_cfg.input_height = config_.det_input_height;
        det_cfg.confidence_threshold = config_.det_conf_thresh;

        detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

TurbineInspector::TurbineInspector(const TurbineConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TurbineInspector::~TurbineInspector() = default;
TurbineInspector::TurbineInspector(TurbineInspector&&) noexcept = default;
TurbineInspector& TurbineInspector::operator=(TurbineInspector&&) noexcept = default;

InspectionResult TurbineInspector::inspect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("TurbineInspector is null.");

    InspectionResult result;
    result.annotated_image = image.clone();

    // --- Step 1: Segment the Blade ---
    preproc::ImageFrame frame{image.data, image.cols, image.rows, preproc::ImageFormat::BGR};

    pimpl_->seg_preproc_->process(frame, pimpl_->seg_in);
    pimpl_->seg_engine_->predict({pimpl_->seg_in}, {pimpl_->seg_out});
    auto seg_res = pimpl_->seg_postproc_->process(pimpl_->seg_out);

    // Get blade mask (assuming class 1 = blade)
    cv::Mat mask;
    cv::inRange(seg_res.mask, cv::Scalar(1), cv::Scalar(1), mask);
    cv::resize(mask, mask, image.size(), 0, 0, cv::INTER_NEAREST);

    // --- Step 2: Create Masked Image ---
    // Zero out the background to help the detector focus
    cv::Mat masked_image = cv::Mat::zeros(image.size(), image.type());
    image.copyTo(masked_image, mask);

    // --- Step 3: Detect Defects on the Masked Image ---
    // The ObjectDetector zoo module handles its own pre/post processing.
    auto detections = pimpl_->detector_->predict(masked_image);

    // --- Step 4: Format Results ---
    for (const auto& det : detections) {
        BladeDefect defect;
        defect.type = det.label;
        defect.confidence = det.confidence;

        // BoundingBox in detector is already scaled to original image
        defect.box = {det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_id};

        result.defects.push_back(defect);

        // Draw
        cv::Rect r((int)det.x1, (int)det.y1, (int)(det.x2-det.x1), (int)(det.y2-det.y1));
        cv::rectangle(result.annotated_image, r, cv::Scalar(0, 0, 255), 2);
        cv::putText(result.annotated_image, det.label, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2);
    }

    // --- Step 5: Health Assessment ---
    if (result.defects.empty()) {
        result.health_score = 1.0f;
        result.requires_maintenance = false;
    } else {
        // Simple heuristic: Score decreases with number of defects
        result.health_score = 1.0f / (1.0f + result.defects.size());
        result.requires_maintenance = true;
    }

    return result;
}

} // namespace xinfer::zoo::energy