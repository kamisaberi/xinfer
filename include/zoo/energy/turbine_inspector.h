#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::energy {

    struct BladeDefect {
        std::string type;     // "Crack", "Erosion", "Lightning"
        float confidence;
        postproc::BoundingBox box; // Location on the blade
    };

    struct InspectionResult {
        std::vector<BladeDefect> defects;

        // Blade health score (0=Damaged, 1=Healthy)
        float health_score;
        bool requires_maintenance;

        // Visualization
        cv::Mat annotated_image;
    };

    struct TurbineConfig {
        // Hardware Target (Drone usually has Jetson/Qualcomm)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Blade Segmenter (UNet) ---
        std::string seg_model_path;
        int seg_input_size = 512;

        // --- Model 2: Damage Detector (YOLO) ---
        std::string det_model_path;
        std::string det_labels_path; // "Crack", "Erosion", etc.
        int det_input_width = 640;
        int det_input_height = 640;
        float det_conf_thresh = 0.4f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TurbineInspector {
    public:
        explicit TurbineInspector(const TurbineConfig& config);
        ~TurbineInspector();

        // Move semantics
        TurbineInspector(TurbineInspector&&) noexcept;
        TurbineInspector& operator=(TurbineInspector&&) noexcept;
        TurbineInspector(const TurbineInspector&) = delete;
        TurbineInspector& operator=(const TurbineInspector&) = delete;

        /**
         * @brief Inspect a turbine blade from an image.
         *
         * @param image Input drone/camera photo.
         * @return Inspection report.
         */
        InspectionResult inspect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::energy