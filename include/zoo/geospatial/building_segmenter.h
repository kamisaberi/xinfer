#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::geospatial {

    struct BuildingResult {
        // Pixel mask of all detected buildings
        // 255 = Building, 0 = Not Building
        cv::Mat footprint_mask;

        // Total area covered by buildings
        float total_building_area_sq_meters;

        // Count of individual building instances
        int building_count;

        // Visualization overlay
        cv::Mat visualization;
    };

    struct BuildingConfig {
        // Hardware Target (High-end GPU for satellite tiles)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_buildings.engine)
        std::string model_path;

        // Input Specs
        int input_width = 1024; // Geospatial models often use larger tiles
        int input_height = 1024;

        // Calibration
        float sq_meters_per_pixel = 0.5f; // From satellite GSD (Ground Sample Distance)

        // Post-processing
        float confidence_threshold = 0.5f; // Binary threshold for mask
        int min_area_pixels = 100;         // Filter out small noise detections

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class BuildingSegmenter {
    public:
        explicit BuildingSegmenter(const BuildingConfig& config);
        ~BuildingSegmenter();

        // Move semantics
        BuildingSegmenter(BuildingSegmenter&&) noexcept;
        BuildingSegmenter& operator=(BuildingSegmenter&&) noexcept;
        BuildingSegmenter(const BuildingSegmenter&) = delete;
        BuildingSegmenter& operator=(const BuildingSegmenter&) = delete;

        /**
         * @brief Segment buildings in an aerial image tile.
         *
         * @param image Input aerial/satellite image.
         * @return Segmentation results and metrics.
         */
        BuildingResult segment(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial