#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::civil {

    /**
     * @brief A detected fault on a grid component.
     */
    struct GridFault {
        std::string component_type; // "Insulator", "Damper"
        std::string fault_type;     // "Crack", "Corrosion", "OK"
        float confidence;
        postproc::BoundingBox box;
    };

    struct GridInspectionResult {
        std::vector<GridFault> faults;
        bool requires_maintenance;

        // Visualization
        cv::Mat annotated_image;
    };

    struct GridConfig {
        // Hardware Target (Drone usually has Jetson/Qualcomm)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Component Detector (YOLO) ---
        std::string detector_path;
        std::string component_labels_path; // "Insulator", "Damper", etc.
        int det_input_width = 640;
        int det_input_height = 640;

        // --- Model 2: Fault Classifier (ResNet) ---
        std::string classifier_path;
        std::string fault_labels_path; // "OK", "Crack", "Corrosion"
        int cls_input_width = 224;
        int cls_input_height = 224;

        // Sensitivity
        float det_conf_thresh = 0.5f;
        float cls_conf_thresh = 0.6f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class GridInspector {
    public:
        explicit GridInspector(const GridConfig& config);
        ~GridInspector();

        // Move semantics
        GridInspector(GridInspector&&) noexcept;
        GridInspector& operator=(GridInspector&&) noexcept;
        GridInspector(const GridInspector&) = delete;
        GridInspector& operator=(const GridInspector&) = delete;

        /**
         * @brief Inspect a power grid image for faults.
         *
         * @param image Input drone/camera photo.
         * @return Inspection report.
         */
        GridInspectionResult inspect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::civil