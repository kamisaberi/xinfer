#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::robotics {

    /**
     * @brief Current state of the robot.
     * Inputs usually normalized from sensors.
     */
    struct RobotState {
        // Joint Space
        std::vector<float> joint_positions; // Rad
        std::vector<float> joint_velocities; // Rad/s

        // End-Effector Space (Optional, depends on model)
        // [x, y, z, qx, qy, qz, qw]
        std::vector<float> ee_pose;

        // Force/Torque Sensor (Critical for assembly)
        // [fx, fy, fz, tx, ty, tz]
        std::vector<float> force_torque;
    };

    /**
     * @brief Control action to be sent to the robot controller.
     */
    struct RobotAction {
        // Usually joint velocities or joint torques
        std::vector<float> commands;
    };

    struct PolicyConfig {
        // Hardware Target
        // For 500Hz+ loops, CPU (AVX) is often preferred over GPU due to PCI latency.
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ppo_insertion.onnx)
        std::string model_path;

        // Input Dimensions
        int num_joints = 6;
        int input_dim = 0; // Calculated or manual override
        int output_dim = 6;

        // Normalization (RL models usually expect inputs in [-1, 1] or N(0,1))
        std::vector<float> state_mean;
        std::vector<float> state_std;

        // Action Scaling (Model output [-1, 1] -> Real limits)
        // e.g., Max joint velocity 0.5 rad/s
        std::vector<float> action_scale;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class AssemblyPolicy {
    public:
        explicit AssemblyPolicy(const PolicyConfig& config);
        ~AssemblyPolicy();

        // Move semantics
        AssemblyPolicy(AssemblyPolicy&&) noexcept;
        AssemblyPolicy& operator=(AssemblyPolicy&&) noexcept;
        AssemblyPolicy(const AssemblyPolicy&) = delete;
        AssemblyPolicy& operator=(const AssemblyPolicy&) = delete;

        /**
         * @brief Calculate the next control action.
         *
         * Designed to be called inside a high-frequency control loop.
         * Allocations are minimized.
         *
         * @param state Current robot observations.
         * @return Action vector.
         */
        RobotAction step(const RobotState& state);

        /**
         * @brief Reset internal policy state (for LSTM/RNN policies).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::robotics