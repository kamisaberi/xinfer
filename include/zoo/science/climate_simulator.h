#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::science {

    /**
     * @brief A state vector representing the atmosphere at a specific time.
     * Flattened vector of size [NumVariables * Height * Width].
     * Layout: Channel-First (CHW).
     */
    struct AtmosphericState {
        std::vector<float> data;
        uint64_t timestamp; // Unix timestamp
    };

    /**
     * @brief Definition of a physical variable (e.g., Temperature at 2m).
     */
    struct VariableDef {
        std::string name;
        float mean; // For normalization
        float std;  // For normalization
    };

    struct ClimateConfig {
        // Hardware Target (High-end GPU recommended for Transformers)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., fourcastnet.engine)
        std::string model_path;

        // Grid Dimensions (Lat/Lon)
        int grid_height = 720; // Latitude
        int grid_width = 1440; // Longitude

        // Variables (Channels)
        // e.g., U10, V10, T2M, MSL (Wind U/V, Temp, Pressure)
        std::vector<VariableDef> variables;

        // Simulation Time Step (e.g., 6 hours per inference step)
        int step_hours = 6;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ClimateSimulator {
    public:
        explicit ClimateSimulator(const ClimateConfig& config);
        ~ClimateSimulator();

        // Move semantics
        ClimateSimulator(ClimateSimulator&&) noexcept;
        ClimateSimulator& operator=(ClimateSimulator&&) noexcept;
        ClimateSimulator(const ClimateSimulator&) = delete;
        ClimateSimulator& operator=(const ClimateSimulator&) = delete;

        /**
         * @brief Run a single simulation step.
         * Predicts state at t + step_hours based on input.
         *
         * @param current_state Raw physical values (Kelvin, Pascals, m/s).
         * @return Predicted state (Denormalized to physical units).
         */
        AtmosphericState step(const AtmosphericState& current_state);

        /**
         * @brief Run an autoregressive forecast (multi-step).
         * Feeds the output of step N as input to step N+1.
         *
         * @param initial_state Starting conditions.
         * @param num_steps How many steps to forecast.
         * @return Trajectory of states.
         */
        std::vector<AtmosphericState> forecast(const AtmosphericState& initial_state, int num_steps);

        /**
         * @brief Reset internal buffers.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::science