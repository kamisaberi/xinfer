#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::space {

    /**
     * @brief A tracked piece of space debris.
     */
    struct DebrisObject {
        int track_id;       // Persistent ID
        float x, y;         // Center coordinates
        float w, h;         // Size
        float confidence;

        // Estimated motion vector (pixels/frame)
        // Critical for orbital estimation
        float velocity_x;
        float velocity_y;
    };

    struct DebrisConfig {
        // Hardware Target (Radiation-tolerant FPGAs preferred)
        xinfer::Target target = xinfer::Target::AMD_VITIS;

        // Model Path (e.g., debris_yolo_tiny.xmodel)
        std::string model_path;

        // Input Specs
        // Space sensors often use square, high-res monochrome inputs
        int input_width = 1024;
        int input_height = 1024;

        // Detection Thresholds
        float conf_threshold = 0.3f; // Debris is faint, lower threshold
        float nms_threshold = 0.1f;  // Objects shouldn't overlap in space

        // Tracking Settings
        int max_track_age = 60;      // Keep tracks alive longer (objects don't disappear)
        int min_hits = 5;            // Wait longer to confirm to reject noise

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DebrisTracker {
    public:
        explicit DebrisTracker(const DebrisConfig& config);
        ~DebrisTracker();

        // Move semantics
        DebrisTracker(DebrisTracker&&) noexcept;
        DebrisTracker& operator=(DebrisTracker&&) noexcept;
        DebrisTracker(const DebrisTracker&) = delete;
        DebrisTracker& operator=(const DebrisTracker&) = delete;

        /**
         * @brief Detect and Track debris in a sensor frame.
         *
         * @param image Input frame (Gray or RGB).
         * @return List of confirmed debris objects.
         */
        std::vector<DebrisObject> track(const cv::Mat& image);

        /**
         * @brief Reset tracking state (e.g., after slew maneuver).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::space