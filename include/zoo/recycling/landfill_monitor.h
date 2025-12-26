#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::recycling {

    /**
     * @brief Breakdown of waste types in the current view.
     */
    struct CompositionStats {
        std::map<std::string, float> material_percentages; // e.g., "Plastic": 20.5%
        float fill_level; // 0.0 - 1.0 (Percentage of designated area covered by waste)
        float total_area_px;
    };

    /**
     * @brief Result of the monitoring scan.
     */
    struct MonitorResult {
        CompositionStats stats;

        // Color-coded segmentation mask for visualization
        // (Green=Organic, Blue=Plastic, Red=Hazard, etc.)
        cv::Mat segmentation_vis;

        bool alert_capacity; // True if fill_level > max_capacity
    };

    struct MonitorConfig {
        // Hardware Target (Edge GPU or NPU recommended for Segmentation)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., unet_waste_material.onnx)
        // Trained to classify pixels into materials.
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Class Mapping
        // Map Model Class Index -> Material Name ("Background", "Plastic", "Metal", etc.)
        std::vector<std::string> class_names;

        // Visualization Colors (R, G, B) per class
        std::vector<std::vector<uint8_t>> class_colors;

        // Thresholds
        float capacity_threshold = 0.85f; // Alert if 85% full

        // Optional: Define a polygon ROI (Region of Interest) for the pit
        // If empty, assumes whole image is the pit.
        std::vector<cv::Point> roi_polygon;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class LandfillMonitor {
    public:
        explicit LandfillMonitor(const MonitorConfig& config);
        ~LandfillMonitor();

        // Move semantics
        LandfillMonitor(LandfillMonitor&&) noexcept;
        LandfillMonitor& operator=(LandfillMonitor&&) noexcept;
        LandfillMonitor(const LandfillMonitor&) = delete;
        LandfillMonitor& operator=(const LandfillMonitor&) = delete;

        /**
         * @brief Analyze a frame of the landfill.
         *
         * Pipeline:
         * 1. Preprocess
         * 2. Semantic Segmentation (Pixel Classification)
         * 3. Mask Statistics (Count pixels per class within ROI)
         *
         * @param image Input camera frame.
         * @return Statistics and visualization.
         */
        MonitorResult analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::recycling