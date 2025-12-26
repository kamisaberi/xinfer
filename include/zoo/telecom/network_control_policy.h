#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::telecom {

    /**
     * @brief Current State of the Network Element.
     * Normalized metrics from the RAN/Core.
     */
    struct NetworkState {
        // Core metrics
        float throughput_mbps;
        float latency_ms;
        float packet_loss_rate; // 0.0 - 1.0
        float queue_depth_bytes;

        // Optional: Custom features (Signal Strength (RSRP), User Count, etc.)
        std::vector<float> extra_features;
    };

    /**
     * @brief Recommended Action.
     */
    struct ControlAction {
        // Continuous actions (e.g., Power Level, Target Bitrate)
        std::vector<float> continuous_actions;

        // Discrete action ID (e.g., Modulation Scheme Index, Handover Target ID)
        int discrete_action_id;
        float action_confidence;
    };

    struct PolicyConfig {
        // Hardware Target
        // Telecom edge usually uses Intel CPU (FlexRAN) or FPGA/ASIC.
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., dqn_scheduler.onnx, ppo_power.rknn)
        std::string model_path;

        // Model Architecture Specs
        // Input: [1, lookback_window, input_features]
        int input_features = 4; // Matches basic NetworkState fields
        int lookback_window = 10; // How many past states to consider

        // Output Specs
        bool is_discrete = false; // True = Classification (ArgMax), False = Regression
        int output_dim = 1;

        // Normalization (State -> Tensor)
        // (x - mean) / std
        std::vector<float> mean;
        std::vector<float> std;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class NetworkController {
    public:
        explicit NetworkController(const PolicyConfig& config);
        ~NetworkController();

        // Move semantics
        NetworkController(NetworkController&&) noexcept;
        NetworkController& operator=(NetworkController&&) noexcept;
        NetworkController(const NetworkController&) = delete;
        NetworkController& operator=(const NetworkController&) = delete;

        /**
         * @brief Ingest a new state and get the next control action.
         *
         * This function updates the internal history buffer and runs inference.
         *
         * @param state Current network metrics.
         * @return The recommended action to apply.
         */
        ControlAction step(const NetworkState& state);

        /**
         * @brief Reset the internal state history (e.g., on UE handover or session reset).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::telecom