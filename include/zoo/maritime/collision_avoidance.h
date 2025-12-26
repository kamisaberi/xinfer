#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::maritime {

    enum class RiskLevel {
        SAFE = 0,
        WARNING = 1,    // Object approaching
        CRITICAL = 2    // Collision imminent
    };

    struct Obstacle {
        int track_id;
        std::string type;     // "Ship", "Buoy", "Iceberg"
        float distance_m;     // Estimated distance in meters
        float bearing_deg;    // Angle relative to bow (-30 to +30)
        float relative_speed; // meters/sec (approaching speed)
        float ttc;            // Time To Collision (seconds)

        postproc::BoundingBox box;
    };

    struct AvoidanceResult {
        RiskLevel status;
        std::vector<Obstacle> obstacles;

        // Navigation suggestion
        float suggested_rudder_angle; // +ve Turn Right, -ve Turn Left
    };

    struct CollisionConfig {
        // Hardware Target (NVIDIA Jetson is common for boats)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., yolo_seaships.engine)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Camera Calibration (For Distance Estimation)
        // Simple Pinhole / Flat-Water assumption
        float focal_length_px = 1000.0f;
        float camera_height_m = 5.0f;    // Height of camera above water line
        float horizon_y_px = 300.0f;     // Y-coordinate of the horizon line

        // Safety Thresholds
        float critical_ttc_sec = 10.0f;  // Alert if impact < 10s
        float warning_distance_m = 100.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CollisionAvoidance {
    public:
        explicit CollisionAvoidance(const CollisionConfig& config);
        ~CollisionAvoidance();

        // Move semantics
        CollisionAvoidance(CollisionAvoidance&&) noexcept;
        CollisionAvoidance& operator=(CollisionAvoidance&&) noexcept;
        CollisionAvoidance(const CollisionAvoidance&) = delete;
        CollisionAvoidance& operator=(const CollisionAvoidance&) = delete;

        /**
         * @brief Process a frame to detect collision risks.
         *
         * Pipeline:
         * 1. Detect Ships/Obstacles.
         * 2. Track to estimate velocity.
         * 3. Estimate distance using Camera Geometry (Flat Plane assumption).
         * 4. Calculate Risk & Maneuver.
         *
         * @param image Camera frame.
         * @param dt_sec Time elapsed since last frame (for velocity calc).
         * @return Situation analysis.
         */
        AvoidanceResult process(const cv::Mat& image, float dt_sec);

        /**
         * @brief Reset tracking state.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::maritime