#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::special {

    /**
     * @brief State vector of a physical entity (e.g., Particle, Fluid Cell).
     */
    struct PhysicalState {
        // Generic feature vector.
        // Example for Particle: [x, y, z, px, py, pz, energy, charge]
        // Example for Fluid: [pressure, velocity_x, velocity_y, density]
        std::vector<float> features;

        // Optional ID for tracking
        int id = 0;
    };

    /**
     * @brief Result of the Physics Inference.
     */
    struct PhysicsResult {
        // Predicted next state (Regression)
        std::vector<float> next_state;

        // Classification (e.g., Particle Type ID)
        int classification_id;
        float confidence;
    };

    struct PhysicsConfig {
        // Hardware Target
        // FPGA (AMD_VITIS / INTEL_FPGA) is very common in Physics triggers.
        xinfer::Target target = xinfer::Target::INTEL_FPGA;

        // Model Path (e.g., particle_transformer.xmodel, flow_net.onnx)
        std::string model_path;

        // Dimensions
        int input_dim = 8;  // Size of PhysicalState::features
        int output_dim = 8; // Size of predicted state

        // Normalization (Standard Scaling: (x - u) / s)
        // Physics data often spans many orders of magnitude (e.g. Energy vs Position).
        // Proper scaling is mandatory.
        std::vector<float> mean;
        std::vector<float> std;

        // Vendor flags (e.g. "LATENCY_OPTIMIZED" for L1 Triggers)
        std::vector<std::string> vendor_params;
    };

    class PhysicsEngine {
    public:
        explicit PhysicsEngine(const PhysicsConfig& config);
        ~PhysicsEngine();

        // Move semantics
        PhysicsEngine(PhysicsEngine&&) noexcept;
        PhysicsEngine& operator=(PhysicsEngine&&) noexcept;
        PhysicsEngine(const PhysicsEngine&) = delete;
        PhysicsEngine& operator=(const PhysicsEngine&) = delete;

        /**
         * @brief Predict the evolution of a physical system.
         *
         * @param current_state Current physical parameters.
         * @return Predicted next state or classification.
         */
        PhysicsResult simulate_step(const PhysicalState& current_state);

        /**
         * @brief Batch simulation (High Throughput).
         * Useful for Monte Carlo simulations with thousands of particles.
         */
        std::vector<PhysicsResult> simulate_batch(const std::vector<PhysicalState>& states);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special