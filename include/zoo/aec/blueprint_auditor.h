#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::aec {

    /**
     * @brief A detected architectural symbol.
     */
    struct ArchSymbol {
        std::string type; // "Door", "Window", "Wall", "Stair"
        postproc::BoundingBox box;
        float confidence;
    };

    enum class ComplianceStatus {
        PASS = 0,
        FAIL = 1,
        WARNING = 2
    };

    struct AuditRuleResult {
        std::string rule_name;
        ComplianceStatus status;
        std::string message;
    };

    struct AuditResult {
        std::vector<ArchSymbol> symbols;
        std::vector<AuditRuleResult> rule_results;
        bool is_compliant;

        // Visualization
        cv::Mat annotated_blueprint;
    };

    struct AuditorConfig {
        // Hardware Target (CPU/iGPU is fine for document analysis)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8_blueprint.onnx)
        std::string model_path;

        // Label Map (Class ID -> Symbol Name)
        std::string labels_path;

        // Input Specs
        int input_width = 1024; // Blueprints are high-res
        int input_height = 1024;

        // Detection Sensitivity
        float conf_threshold = 0.6f;
        float nms_threshold = 0.4f;

        // Rule Engine Parameters
        float min_hallway_width_m = 0.9f;
        float pixels_per_meter = 100.0f; // Calibration

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class BlueprintAuditor {
    public:
        explicit BlueprintAuditor(const AuditorConfig& config);
        ~BlueprintAuditor();

        // Move semantics
        BlueprintAuditor(BlueprintAuditor&&) noexcept;
        BlueprintAuditor& operator=(BlueprintAuditor&&) noexcept;
        BlueprintAuditor(const BlueprintAuditor&) = delete;
        BlueprintAuditor& operator=(const BlueprintAuditor&) = delete;

        /**
         * @brief Audit a blueprint image.
         *
         * @param image Input image of the blueprint.
         * @return The full audit report.
         */
        AuditResult audit(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::aec