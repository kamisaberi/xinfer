#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::drones {

    /**
     * @brief Low-level control commands for the drone flight controller.
     */
    struct FlightCommand {
        // Linear Velocities (m/s) in drone's body frame
        float velocity_forward; // X+
        float velocity_right;   // Y+
        float velocity_up;      // Z+

        // Angular Velocity (rad/s)
        float yaw_rate;
    };

    struct NavPolicyConfig {
        // Hardware Target (Jetson/Qualcomm Robotics are standard)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., drone_ppo_nav.engine)
        // Expected Input: [1, C, H, W] (Image)
        // Expected Output: [1, 4] (Action Vector)
        std::string model_path;

        // Input Specs
        int input_width = 224;
        int input_height = 224;

        // Action Scaling
        // Model outputs are usually [-1, 1], we scale them to physical limits
        float max_linear_velocity = 2.0f; // m/s
        float max_angular_velocity = 1.5f; // rad/s

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class NavigationPolicy {
    public:
        explicit NavigationPolicy(const NavPolicyConfig& config);
        ~NavigationPolicy();

        // Move semantics
        NavigationPolicy(NavigationPolicy&&) noexcept;
        NavigationPolicy& operator=(NavigationPolicy&&) noexcept;
        NavigationPolicy(const NavigationPolicy&) = delete;
        NavigationPolicy& operator=(const NavigationPolicy&) = delete;

        /**
         * @brief Compute the next flight command from a camera frame.
         *
         * @param image Input camera view.
         * @return The control action to send to the flight controller.
         */
        FlightCommand get_action(const cv::Mat& image);

        /**
         * @brief Reset internal state (for RNN-based policies).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::drones