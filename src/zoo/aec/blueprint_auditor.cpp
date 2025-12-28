#include <xinfer/zoo/aec/blueprint_auditor.h>
#include <xinfer/core/logging.h>

// --- Reuse the generic Object Detector ---
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <vector>
#include <string>

namespace xinfer::zoo::aec {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct BlueprintAuditor::Impl {
    AuditorConfig config_;

    // High-level Zoo module for detection
    std::unique_ptr<vision::ObjectDetector> detector_;

    Impl(const AuditorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the Object Detector for this task
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

    // --- Rule Engine ---
    void run_rules(const std::vector<ArchSymbol>& symbols, AuditResult& result) {
        bool all_pass = true;

        // Rule 1: Check hallway width
        // Heuristic: Find pairs of 'Wall' symbols that are parallel and close
        // (This is a complex geometric problem, simplified here)
        // A simpler rule could be to find "Room" segmentations and check door counts.
        // Let's assume a "Hallway" class exists for this rule.

        for (const auto& sym : symbols) {
            if (sym.type == "Hallway") {
                // Check if width is compliant
                float width_px = std::min(sym.box.x2 - sym.box.x1, sym.box.y2 - sym.box.y1);
                float width_m = width_px / config_.pixels_per_meter;

                if (width_m < config_.min_hallway_width_m) {
                    result.rule_results.push_back({
                        "Hallway Width", ComplianceStatus::FAIL,
                        "Hallway at (" + std::to_string((int)sym.box.x1) + "," + std::to_string((int)sym.box.y1) +
                        ") is " + std::to_string(width_m) + "m wide, requires " + std::to_string(config_.min_hallway_width_m) + "m."
                    });
                    all_pass = false;
                }
            }
        }

        // Rule 2: Every bedroom must have a window
        // (This requires Room segmentation, a more advanced model)

        // If all rules passed
        if (all_pass) {
             result.rule_results.push_back({"All Checks", ComplianceStatus::PASS, "Blueprint meets basic checks."});
        }

        result.is_compliant = all_pass;
    }
};

// =================================================================================
// Public API
// =================================================================================

BlueprintAuditor::BlueprintAuditor(const AuditorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

BlueprintAuditor::~BlueprintAuditor() = default;
BlueprintAuditor::BlueprintAuditor(BlueprintAuditor&&) noexcept = default;
BlueprintAuditor& BlueprintAuditor::operator=(BlueprintAuditor&&) noexcept = default;

AuditResult BlueprintAuditor::audit(const cv::Mat& image) {
    if (!pimpl_ || !pimpl_->detector_) {
        throw std::runtime_error("BlueprintAuditor is not initialized.");
    }

    AuditResult result;
    result.annotated_blueprint = image.clone();

    // 1. Detect all architectural symbols
    auto detections = pimpl_->detector_->predict(image);

    // 2. Map to ArchSymbol struct
    for (const auto& det : detections) {
        ArchSymbol sym;
        sym.type = det.label;
        sym.confidence = det.confidence;
        sym.box = {det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_id};
        result.symbols.push_back(sym);

        // Draw on image
        cv::Rect r((int)det.x1, (int)det.y1, (int)(det.x2 - det.x1), (int)(det.y2 - det.y1));
        cv::rectangle(result.annotated_blueprint, r, cv::Scalar(255, 0, 0), 2);
        cv::putText(result.annotated_blueprint, det.label, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,0,0), 2);
    }

    // 3. Run Rule Engine
    pimpl_->run_rules(result.symbols, result);

    return result;
}

} // namespace xinfer::zoo::aec