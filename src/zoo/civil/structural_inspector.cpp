#include <xinfer/zoo/civil/structural_inspector.h>
#include <xinfer/core/logging.h>

// --- We reuse the generic Object Detector module ---
#include <xinfer/zoo/vision/detector.h>

#include <iostream>

namespace xinfer::zoo::civil {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct StructuralInspector::Impl {
    InspectorConfig config_;

    // High-level Zoo module for detection
    std::unique_ptr<vision::ObjectDetector> detector_;

    Impl(const InspectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Configure the Object Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.model_path;
        det_cfg.labels_path = config_.labels_path;
        det_cfg.input_width = config_.input_width;
        det_cfg.input_height = config_.input_height;
        det_cfg.confidence_threshold = config_.conf_threshold;
        det_cfg.nms_iou_threshold = config_.nms_threshold;
        det_cfg.vendor_params = config_.vendor_params;

        detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);
    }

    // --- Core Logic: Classify Severity ---
    DefectSeverity determine_severity(const std::string& type, const postproc::BoundingBox& box) {
        // Simple rule-based system.
        // In a real application, this could be a secondary ML model.

        // Severe defects
        if (type == "Exposed_Rebar" || type == "Deep_Crack" || type == "Structural_Damage") {
            return DefectSeverity::HIGH;
        }

        // Medium severity
        float area = (box.x2 - box.x1) * (box.y2 - box.y1);
        if (type == "Spalling" || (type == "Crack" && area > 2000)) { // Large crack
            return DefectSeverity::MEDIUM;
        }

        // Low severity
        return DefectSeverity::LOW;
    }
};

// =================================================================================
// Public API
// =================================================================================

StructuralInspector::StructuralInspector(const InspectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

StructuralInspector::~StructuralInspector() = default;
StructuralInspector::StructuralInspector(StructuralInspector&&) noexcept = default;
StructuralInspector& StructuralInspector::operator=(StructuralInspector&&) noexcept = default;

InspectionReport StructuralInspector::inspect(const cv::Mat& image) {
    if (!pimpl_ || !pimpl_->detector_) {
        throw std::runtime_error("StructuralInspector is not initialized.");
    }

    InspectionReport result;
    result.annotated_image = image.clone();
    result.requires_attention = false;
    result.max_severity = DefectSeverity::LOW;

    // 1. Run Detection
    auto detections = pimpl_->detector_->predict(image);

    // 2. Analyze & Format Results
    for (const auto& det : detections) {
        StructuralDefect defect;
        defect.type = det.label;
        defect.confidence = det.confidence;
        defect.box = {det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_id};

        // Determine severity
        defect.severity = pimpl_->determine_severity(det.label, defect.box);

        // Update overall report
        if (defect.severity > DefectSeverity::LOW) {
            result.requires_attention = true;
        }
        if (defect.severity > result.max_severity) {
            result.max_severity = defect.severity;
        }

        result.defects.push_back(defect);

        // Visualization
        cv::Scalar color;
        switch(defect.severity) {
            case DefectSeverity::HIGH:   color = cv::Scalar(0, 0, 255); break; // Red
            case DefectSeverity::MEDIUM: color = cv::Scalar(0, 255, 255); break; // Yellow
            case DefectSeverity::LOW:    color = cv::Scalar(255, 0, 0); break; // Blue
        }

        cv::Rect r((int)det.x1, (int)det.y1, (int)(det.x2 - det.x1), (int)(det.y2 - det.y1));
        cv::rectangle(result.annotated_image, r, color, 2);
        cv::putText(result.annotated_image, det.label, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 2);
    }

    return result;
}

} // namespace xinfer::zoo::civil