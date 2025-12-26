#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::robotics {

    /**
     * @brief A single particle in the simulation.
     */
    struct Particle {
        float x, y, z;       // Position
        float vx, vy, vz;    // Velocity
        int material_id;     // 0=Fluid, 1=Elastic, 2=Rigid Boundary
    };

    /**
     * @brief External interaction (e.g., a Gripper pushing the object).
     */
    struct BoundaryCondition {
        float x, y, z;       // Position of the effector
        float vx, vy, vz;    // Velocity of the effector
    };

    struct SimulatorConfig {
        // Hardware Target (GNNs run best on GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., gns_tissue.onnx)
        std::string model_path;

        // Physics Parameters
        float dt = 0.01f;           // Time step size (seconds)
        float connectivity_radius = 0.05f; // Distance to form edges between particles
        float gravity_y = -9.8f;

        // Limits
        int max_particles = 2000;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SoftBodySimulator {
    public:
        explicit SoftBodySimulator(const SimulatorConfig& config);
        ~SoftBodySimulator();

        // Move semantics
        SoftBodySimulator(SoftBodySimulator&&) noexcept;
        SoftBodySimulator& operator=(SoftBodySimulator&&) noexcept;
        SoftBodySimulator(const SoftBodySimulator&) = delete;
        SoftBodySimulator& operator=(const SoftBodySimulator&) = delete;

        /**
         * @brief Initialize the system state.
         */
        void set_state(const std::vector<Particle>& particles);

        /**
         * @brief Advance the simulation by one time step.
         *
         * Pipeline:
         * 1. Neighborhood Search (Build Graph).
         * 2. Neural Inference (Predict Accelerations).
         * 3. Euler Integration (Update Pos/Vel).
         *
         * @param effector Current state of the robot/gripper interacting with the body.
         * @return Updated particle states.
         */
        std::vector<Particle> step(const BoundaryCondition& effector);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::robotics