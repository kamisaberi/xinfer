#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::geospatial {

    struct RoadResult {
        // Binary mask of the road surface (255=Road, 0=Not Road)
        cv::Mat road_mask;

        // Calculated drivable area in sq. meters (if calibrated)
        float drivable_area_sqm;

        // Path of the center of the lane ahead of the vehicle
        std::vector<cv::Point2f> lane_centerline;

        // Visualization
        cv::Mat overlay;
    };

    struct RoadConfig {
        // Hardware Target (NVIDIA Drive, Mobileye, or other Automotive SoC)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_road.engine)
        std::string model_path;

        // Input Specs
        int input_width = 800; // Common for automotive cameras
        int input_height = 288;

        // Class Mapping
        // In the segmentation mask, which class ID represents "Road"?
        int road_class_id = 1;

        // Calibration
        float sq_meters_per_pixel = 0.1f; // Simplified inverse-perspective mapping

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class RoadExtractor {
    public:
        explicit RoadExtractor(const RoadConfig& config);
        ~RoadExtractor();

        // Move semantics
        RoadExtractor(RoadExtractor&&) noexcept;
        RoadExtractor& operator=(RoadExtractor&&) noexcept;
        RoadExtractor(const RoadExtractor&) = delete;
        RoadExtractor& operator=(const RoadExtractor&) = delete;

        /**
         * @brief Extract road information from a frame.
         *
         * @param image Input camera image.
         * @return Road segmentation and path.
         */
        RoadResult extract(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial