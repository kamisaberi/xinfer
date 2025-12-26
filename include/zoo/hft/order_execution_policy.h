#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::hft {

    /**
     * @brief Current state of the trading environment.
     */
    struct MarketState {
        // From Order Book
        // Flattened: [BidP1, BidV1, AskP1, AskV1, BidP2, ...]
        std::vector<float> order_book_features;

        // From Agent
        float time_remaining_norm; // [0.0 - 1.0] (How much time is left to execute)
        float inventory_remaining_norm; // [0.0 - 1.0] (How much of the order is left)
    };

    /**
     * @brief Decision from the policy.
     */
    struct ExecutionDecision {
        // Portion of the remaining inventory to trade in this step.
        // 0.0 = Do nothing
        // 0.1 = Trade 10%
        // 1.0 = Trade everything now
        float trade_volume_ratio;

        // Price level to place the order at (e.g. 0=Best Bid/Ask, 1=Second level)
        int price_level;
    };

    struct ExecutionConfig {
        // Hardware Target (FPGA is best, GPU is second)
        xinfer::Target target = xinfer::Target::AMD_VITIS;

        // Model Path (e.g., ppo_execution.xmodel)
        std::string model_path;

        // Input Specs
        int lookback_window = 50; // Use last 50 market ticks
        int book_depth = 10;
        int num_features = 0; // Calculated

        // Normalization (Mean/Std of market features)
        std::vector<float> mean;
        std::vector<float> std;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class OrderExecutionPolicy {
    public:
        explicit OrderExecutionPolicy(const ExecutionConfig& config);
        ~OrderExecutionPolicy();

        // Move semantics
        OrderExecutionPolicy(OrderExecutionPolicy&&) noexcept;
        OrderExecutionPolicy& operator=(OrderExecutionPolicy&&) noexcept;
        OrderExecutionPolicy(const OrderExecutionPolicy&) = delete;
        OrderExecutionPolicy& operator=(const OrderExecutionPolicy&) = delete;

        /**
        * @brief Reset the policy for a new parent order.
        *
        * @param total_volume The total size of the order to execute.
        * @param total_time_sec The total time horizon for execution.
        */
        void start_new_order(float total_volume, float total_time_sec);

        /**
         * @brief Get the next trade decision.
         *
         * @param current_book The current L2 order book snapshot.
         * @return The next action to take.
         */
        ExecutionDecision get_action(const std::vector<float>& current_book_features);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::hft