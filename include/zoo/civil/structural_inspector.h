#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::civil {

    enum class DefectSeverity {
        LOW,    // Minor cosmetic (e.g., small efflorescence)
        MEDIUM, // Monitor (e.g., hairline crack, minor spalling)
        HIGH    // Immediate Attention (e.g., large crack, exposed rebar)
    };

    struct StructuralDefect {
        std::string type;       // "Crack", "Spalling", "Corrosion"
        DefectSeverity severity;
        float confidence;
        postproc::BoundingBox box;
    };

    struct InspectionReport {
        std::vector<StructuralDefect> defects;

        // Overall assessment
        bool requires_attention;
        DefectSeverity max_severity;

        // Visualization
        cv::Mat annotated_image;
    };

    struct InspectorConfig {
        // Hardware Target (Drone usually has Jetson, ground is often CPU/Intel)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8_concrete_defects.rknn)
        std::string model_path;

        // Label map for the detector
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.4f;
        float nms_threshold = 0.3f; // Cracks can be long and thin, allow some overlap

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class StructuralInspector {
    public:
        explicit StructuralInspector(const InspectorConfig& config);
        ~StructuralInspector();

        // Move semantics
        StructuralInspector(StructuralInspector&&) noexcept;
        StructuralInspector& operator=(Structural-Inspector&&) noexcept;
        StructuralInspector(const StructuralInspector&) = delete;
        StructuralInspector& operator=(const StructuralInspector&) = delete;

        /**
         * @brief Inspect a structural image for defects.
         *
         * @param image Input photo (from drone or ground inspection).
         * @return Inspection report.
         */
        InspectionReport inspect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::civil