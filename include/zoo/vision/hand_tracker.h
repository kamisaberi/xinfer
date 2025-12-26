#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Hand Tracking.
     */
    struct HandResult {
        int track_id;       // Persistent ID
        postproc::BoundingBox box;

        // Future extension: std::vector<Point> landmarks;
        // (Requires a Pose Postprocessor)
    };

    struct HandTrackerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8n-hand.engine)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Thresholds
        float conf_threshold = 0.5f;
        float nms_threshold = 0.5f;

        // Tracking Settings
        int max_track_age = 15; // Hands move fast; drop tracks quickly if lost
        int min_hits = 2;       // Confirm track quickly

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class HandTracker {
    public:
        explicit HandTracker(const HandTrackerConfig& config);
        ~HandTracker();

        // Move semantics
        HandTracker(HandTracker&&) noexcept;
        HandTracker& operator=(HandTracker&&) noexcept;
        HandTracker(const HandTracker&) = delete;
        HandTracker& operator=(const HandTracker&) = delete;

        /**
         * @brief Detect and Track hands in a frame.
         *
         * @param image Input video frame.
         * @return List of tracked hands with IDs.
         */
        std::vector<HandResult> track(const cv::Mat& image);

        /**
         * @brief Reset tracking state.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision