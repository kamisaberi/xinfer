#include <xinfer/zoo/special/hft.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Pre/Post factories skipped for custom low-latency math path.

#include <iostream>
#include <deque>
#include <cmath>
#include <algorithm>
#include <cstring>
#include <chrono>

namespace xinfer::zoo::special {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct HftStrategy::Impl {
    HftConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Feature Buffer
    // Stores flattened features for each tick
    // Features per tick = Depth * 4 (BidP, BidV, AskP, AskV)
    std::deque<std::vector<float>> history_buffer_;

    // Cached previous mid-price for log-return calculation
    float prev_mid_price_ = 0.0f;
    int num_features_per_tick_ = 0;

    Impl(const HftConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Calculate Feature Size
        // 4 arrays (BidP, BidV, AskP, AskV) * Depth
        num_features_per_tick_ = config_.book_depth * 4;

        // 2. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("HftStrategy: Failed to load model " + config_.model_path);
        }

        // 3. Pre-allocate Input Tensor
        // Shape: [1, Lookback, Features] (Sequence Model) or [1, Features, Lookback] (1D-CNN)
        // Assuming [1, Lookback, Features] (Batch-First)
        input_tensor.resize({1, (int64_t)config_.lookback_window, (int64_t)num_features_per_tick_}, core::DataType::kFLOAT);

        reset();
    }

    void reset() {
        history_buffer_.clear();
        prev_mid_price_ = 0.0f;

        // Pad buffer with zeros to allow immediate inference
        std::vector<float> zeros(num_features_per_tick_, 0.0f);
        for(int i=0; i<config_.lookback_window; ++i) {
            history_buffer_.push_back(zeros);
        }
    }

    // --- Feature Engineering (Critical Path) ---
    // Converts raw Order Book -> Normalized Vector
    std::vector<float> process_tick(const OrderBookSnapshot& snap) {
        std::vector<float> feats;
        feats.reserve(num_features_per_tick_);

        // Calculate Mid Price for normalization
        float best_bid = snap.bid_prices.empty() ? 0.0f : snap.bid_prices[0];
        float best_ask = snap.ask_prices.empty() ? 0.0f : snap.ask_prices[0];
        float mid_price = (best_bid + best_ask) * 0.5f;

        if (mid_price < 1e-6) mid_price = 1.0f; // Safety

        // 1. Prices (Normalize via Log Return relative to Mid Price)
        // Or normalize relative to best bid/ask
        for(float p : snap.bid_prices) {
            // (Price - Mid) / Mid ~= Log Return if diff is small
            feats.push_back((p - mid_price) / mid_price);
        }
        for(float v : snap.bid_vols) {
            // Log volume to squash spikes
            feats.push_back(std::log1p(v));
        }
        for(float p : snap.ask_prices) {
            feats.push_back((p - mid_price) / mid_price);
        }
        for(float v : snap.ask_vols) {
            feats.push_back(std::log1p(v));
        }

        // Pad if snapshot was smaller than depth config
        while(feats.size() < (size_t)num_features_per_tick_) {
            feats.push_back(0.0f);
        }

        return feats;
    }

    void update_tensor() {
        // Flatten Deque into Contiguous Tensor Memory
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        for (const auto& tick_feats : history_buffer_) {
            std::memcpy(ptr + idx, tick_feats.data(), num_features_per_tick_ * sizeof(float));
            idx += num_features_per_tick_;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

HftStrategy::HftStrategy(const HftConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

HftStrategy::~HftStrategy() = default;
HftStrategy::HftStrategy(HftStrategy&&) noexcept = default;
HftStrategy& HftStrategy::operator=(HftStrategy&&) noexcept = default;

void HftStrategy::reset() {
    if (pimpl_) pimpl_->reset();
}

TradeSignal HftStrategy::on_tick(const OrderBookSnapshot& snapshot) {
    if (!pimpl_) throw std::runtime_error("HftStrategy is null.");

    // 1. Feature Extraction (CPU)
    // In a full FPGA deployment, this logic moves to PL (Programmable Logic)
    auto features = pimpl_->process_tick(snapshot);

    // 2. Update Rolling Buffer
    pimpl_->history_buffer_.pop_front();
    pimpl_->history_buffer_.push_back(features);

    // 3. Prepare Tensor
    // Memory copy overhead here. For ultra-low latency, use Zero-Copy / Ring Buffer tensors.
    pimpl_->update_tensor();

    // 4. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 5. Post-process (Decision Logic)
    // Assume output is [1, 3] Logits (Hold, Buy, Sell)
    const float* logits = static_cast<const float*>(pimpl_->output_tensor.data());

    // Softmax
    float exp_hold = std::exp(logits[0]);
    float exp_buy  = std::exp(logits[1]);
    float exp_sell = std::exp(logits[2]);
    float sum = exp_hold + exp_buy + exp_sell;

    float p_hold = exp_hold / sum;
    float p_buy  = exp_buy  / sum;
    float p_sell = exp_sell / sum;

    TradeSignal signal;
    signal.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (p_buy > pimpl_->config_.signal_threshold && p_buy > p_sell) {
        signal.action = TradeAction::BUY;
        signal.confidence = p_buy;
    } else if (p_sell > pimpl_->config_.signal_threshold && p_sell > p_buy) {
        signal.action = TradeAction::SELL;
        signal.confidence = p_sell;
    } else {
        signal.action = TradeAction::HOLD;
        signal.confidence = p_hold;
    }

    // Optional: If model has 2nd output for price regression
    signal.predicted_price_delta = 0.0f;

    return signal;
}

} // namespace xinfer::zoo::special