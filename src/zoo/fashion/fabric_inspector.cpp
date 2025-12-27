#include <xinfer/zoo/fashion/fabric_inspector.h>
#include <xinfer/core/logging.h>

// --- We reuse the AnomalyDetector module for the core logic ---
#include <xinfer/zoo/vision/anomaly_detector.h>

#include <iostream>

namespace xinfer::zoo::fashion {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FabricInspector::Impl {
    FabricConfig config_;

    // --- We use the AnomalyDetector as our main engine ---
    // This is a great example of composing Zoo modules.
    std::unique_ptr<vision::AnomalyDetector> anomaly_detector_;

    Impl(const FabricConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // The FabricInspector is a specialized wrapper around the AnomalyDetector.
        // We configure the AnomalyDetector with the parameters passed to us.

        if (config_.model_type == "Anomaly") {
            vision::AnomalyConfig anom_cfg;
            anom_cfg.target = config_.target;
            anom_cfg.model_path = config_.model_path;
            anom_cfg.input_width = config_.input_width;
            anom_cfg.input_height = config_.input_height;
            anom_cfg.threshold = config_.anomaly_threshold;
            anom_cfg.use_smoothing = true; // Smoothing is good for fabric textures
            anom_cfg.vendor_params = config_.vendor_params;

            anomaly_detector_ = std::make_unique<vision::AnomalyDetector>(anom_cfg);
        } else {
            // Placeholder for a YOLO-based defect detector
            // detector_ = std::make_unique<vision::ObjectDetector>(...);
            XINFER_LOG_WARN("Detection model type not yet implemented for FabricInspector. Using Anomaly mode.");
            // Force anomaly mode or throw error
        }
    }

    // --- Core Logic: Analyze Anomaly Mask ---
    InspectionResult analyze_result(const vision::AnomalyResult& anom_res, const cv::Mat& original_image) {
        InspectionResult result;
        result.is_defective = anom_res.is_anomaly;

        // If defective, find the individual defect blobs
        if (result.is_defective) {
            // Find contours in the binary segmentation mask
            std::vector<std::vector<cv::Point>> contours;
            cv::findContours(anom_res.segmentation, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

            for (const auto& cnt : contours) {
                if (cv::contourArea(cnt) > 20) { // Min pixel area
                    FabricDefect defect;
                    defect.bounding_box = cv::boundingRect(cnt);
                    defect.type = "Anomaly"; // Generic for AE model

                    // Score is the max value in the heatmap under this contour
                    cv::Mat roi = anom_res.heatmap(defect.bounding_box);
                    double min_val, max_val;
                    cv::minMaxLoc(roi, &min_val, &max_val);
                    defect.confidence = (float)max_val;

                    result.defects.push_back(defect);
                }
            }
        }

        // Create visualization
        result.annotated_image = original_image.clone();
        if (result.is_defective) {
            // Blend heatmap onto the image
            cv::Mat blended;
            cv::addWeighted(result.annotated_image, 0.6, anom_res.heatmap, 0.4, 0.0, blended);
            result.annotated_image = blended;

            // Draw boxes
            for (const auto& def : result.defects) {
                cv::rectangle(result.annotated_image, def.bounding_box, cv::Scalar(0, 0, 255), 2);
            }
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

FabricInspector::FabricInspector(const FabricConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FabricInspector::~FabricInspector() = default;
FabricInspector::FabricInspector(FabricInspector&&) noexcept = default;
FabricInspector& FabricInspector::operator=(FabricInspector&&) noexcept = default;

InspectionResult FabricInspector::inspect(const cv::Mat& image) {
    if (!pimpl_ || !pimpl_->anomaly_detector_) {
        throw std::runtime_error("FabricInspector is not properly initialized.");
    }

    // 1. Run the underlying Anomaly Detector
    auto anom_res = pimpl_->anomaly_detector_->inspect(image);

    // 2. Analyze the results to find specific defects
    return pimpl_->analyze_result(anom_res, image);
}

} // namespace xinfer::zoo::fashion