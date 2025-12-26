#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::special {

    enum class TradeAction {
        HOLD = 0,
        BUY = 1,
        SELL = 2
    };

    /**
     * @brief Result of the Strategy Inference.
     */
    struct TradeSignal {
        TradeAction action;
        float confidence;     // Probability of the action
        float predicted_price_delta; // Optional: Predicted price change
        uint64_t timestamp_us; // Timestamp of signal generation
    };

    /**
     * @brief A single snapshot of the Order Book.
     */
    struct OrderBookSnapshot {
        // [Price, Volume] pairs for top N levels
        // E.g., for depth=10, vectors have size 10
        std::vector<float> bid_prices;
        std::vector<float> bid_vols;
        std::vector<float> ask_prices;
        std::vector<float> ask_vols;
        uint64_t timestamp_us;
    };

    struct HftConfig {
        // Hardware Target
        // Recommended: AMD_VITIS (FPGA) or NVIDIA_TRT (GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., alpha_net.engine, lob_transformer.xmodel)
        std::string model_path;

        // Input Specs
        int book_depth = 10;   // Top 10 levels
        int lookback_window = 100; // History of 100 ticks

        // Normalization (Z-Score or Log-Return)
        bool use_log_returns = true; // Use ln(pt / pt-1) instead of raw price

        // Execution Thresholds
        float signal_threshold = 0.7f; // Confidence required to trigger trade

        // Vendor flags (e.g., "LATENCY_OPTIMIZED=TRUE")
        std::vector<std::string> vendor_params;
    };

    class HftStrategy {
    public:
        explicit HftStrategy(const HftConfig& config);
        ~HftStrategy();

        // Move semantics
        HftStrategy(HftStrategy&&) noexcept;
        HftStrategy& operator=(HftStrategy&&) noexcept;
        HftStrategy(const HftStrategy&) = delete;
        HftStrategy& operator=(const HftStrategy&) = delete;

        /**
         * @brief Ingest a new Market Tick/Snapshot and get a signal.
         *
         * @param snapshot Current state of the Order Book.
         * @return TradeSignal (Action).
         */
        TradeSignal on_tick(const OrderBookSnapshot& snapshot);

        /**
         * @brief Reset internal state (e.g., start of trading day).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::special