#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::fashion {

    struct FabricDefect {
        std::string type;     // "Stain", "Pull", "Weave_Error"
        float confidence;
        cv::Rect bounding_box;
    };

    struct InspectionResult {
        bool is_defective;
        std::vector<FabricDefect> defects;

        // Visualization overlay
        cv::Mat annotated_image;
    };

    struct FabricConfig {
        // Hardware Target (Edge CPU/NPU for factory floor cameras)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., fabric_anomaly_ae.onnx)
        std::string model_path;

        // Model type: "Anomaly" (Reconstruction) or "Detection" (YOLO)
        std::string model_type = "Anomaly";

        // Input Specs (High resolution for fine details)
        int input_width = 512;
        int input_height = 512;

        // Sensitivity Thresholds
        float anomaly_threshold = 0.3f; // For AE models
        float detection_threshold = 0.5f; // For YOLO models

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FabricInspector {
    public:
        explicit FabricInspector(const FabricConfig& config);
        ~FabricInspector();

        // Move semantics
        FabricInspector(FabricInspector&&) noexcept;
        FabricInspector& operator=(FabricInspector&&) noexcept;
        FabricInspector(const FabricInspector&) = delete;
        FabricInspector& operator=(const FabricInspector&) = delete;

        /**
         * @brief Inspect a patch of fabric for defects.
         *
         * @param image Input image from the inspection camera.
         * @return Inspection report.
         */
        InspectionResult inspect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::fashion