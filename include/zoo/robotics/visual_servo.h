#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::robotics {

    /**
     * @brief Velocity Command output.
     * Coordinate frame depends on robot mapping (usually camera frame).
     */
    struct ServoCommand {
        float vx; // Linear X (Forward/Back)
        float vy; // Linear Y (Left/Right)
        float vz; // Linear Z (Up/Down)
        float yaw_rate; // Angular (Rotate Left/Right)

        bool target_acquired; // True if object is visible
    };

    struct VisualServoConfig {
        // Hardware Target (Low latency is critical for control loops)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolo_person.engine)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Target Definition
        int target_class_id = 0;   // e.g. 0 = Person, 39 = Bottle
        float conf_threshold = 0.5f;

        // Control Parameters (PID / Proportional)
        float kp_linear = 0.5f;    // Gain for X/Y movement
        float kp_angular = 1.0f;   // Gain for Rotation
        float kp_depth = 1.0f;     // Gain for Forward/Back approach

        // Desired State
        float target_area_ratio = 0.3f; // Stop when object fills 30% of screen (Depth proxy)
        float deadband = 0.05f;         // Ignore errors smaller than 5% of screen size

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class VisualServo {
    public:
        explicit VisualServo(const VisualServoConfig& config);
        ~VisualServo();

        // Move semantics
        VisualServo(VisualServo&&) noexcept;
        VisualServo& operator=(VisualServo&&) noexcept;
        VisualServo(const VisualServo&) = delete;
        VisualServo& operator=(const VisualServo&) = delete;

        /**
         * @brief Compute control command from visual input.
         *
         * Pipeline:
         * 1. Inference: Find target box.
         * 2. Error Calc: Compare box center to image center.
         * 3. Control Law: Generate velocity proportional to error.
         *
         * @param image Input camera frame.
         * @return Velocities to send to robot controller.
         */
        ServoCommand update(const cv::Mat& image);

        /**
         * @brief Update the target class ID at runtime.
         * (e.g. Switch from following "Person" to following "Ball").
         */
        void set_target_id(int class_id);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::robotics