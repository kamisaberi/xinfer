#include <xinfer/zoo/hft/order_execution_policy.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>

#include <iostream>
#include <deque>
#include <cmath>
#include <algorithm>

namespace xinfer::zoo::hft {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct OrderExecutionPolicy::Impl {
    ExecutionConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Tensors
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Agent State
    float total_order_volume_ = 0.0f;
    float remaining_volume_ = 0.0f;
    float total_time_horizon_sec_ = 0.0f;
    float time_elapsed_sec_ = 0.0f;
    float dt_sec_ = 0.1f; // Assumed tick interval

    // Market Data History
    // Each entry is a flattened vector of normalized L2 book data
    std::deque<std::vector<float>> history_buffer_;
    int features_per_tick_ = 0;

    Impl(const ExecutionConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("OrderExecutionPolicy: Failed to load model.");
        }

        // 2. Calculate Feature Size
        features_per_tick_ = config_.book_depth * 4; // BidP, BidV, AskP, AskV

        // Total input = (Market Features + Agent State Features) * Lookback
        // Assuming Agent state is concatenated to each tick's features
        int total_input_dim = features_per_tick_ + 2; // Time + Inventory

        // Update config for tensor allocation
        config_.num_features = total_input_dim;

        // 3. Allocate Tensors
        // Shape: [1, Lookback, TotalFeatures]
        input_tensor.resize({1, (int64_t)config_.lookback_window, (int64_t)config_.num_features}, core::DataType::kFLOAT);
    }

    void reset(float total_volume, float total_time) {
        total_order_volume_ = total_volume;
        remaining_volume_ = total_volume;
        total_time_horizon_sec_ = total_time;
        time_elapsed_sec_ = 0.0f;

        history_buffer_.clear();
        // Pad history with zeros
        std::vector<float> zeros(features_per_tick_, 0.0f);
        for (int i = 0; i < config_.lookback_window; ++i) {
            history_buffer_.push_back(zeros);
        }
    }

    void prepare_input(const std::vector<float>& current_book) {
        // 1. Update history
        history_buffer_.pop_front();
        history_buffer_.push_back(current_book);

        // 2. Flatten and Normalize
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;

        for (const auto& tick_feats : history_buffer_) {
            // A. Market Features
            for (int i = 0; i < features_per_tick_; ++i) {
                float val = tick_feats[i];
                // Normalize using pre-calculated mean/std
                if (i < config_.mean.size()) {
                    val = (val - config_.mean[i]) / (config_.std[i] + 1e-9f);
                }
                ptr[idx++] = val;
            }

            // B. Agent State Features
            // Normalized Time Remaining
            float time_norm = 1.0f - (time_elapsed_sec_ / total_time_horizon_sec_);
            ptr[idx++] = std::max(0.0f, time_norm);

            // Normalized Inventory Remaining
            float inv_norm = remaining_volume_ / total_order_volume_;
            ptr[idx++] = inv_norm;
        }
    }

    ExecutionDecision decode_output() {
        ExecutionDecision dec;
        const float* out = static_cast<const float*>(output_tensor.data());

        // Assuming model outputs 2 values:
        // Output[0]: Volume Ratio (0..1) -> often via Sigmoid
        // Output[1]: Price Level Logits (ArgMax)

        dec.trade_volume_ratio = out[0]; // Already in [0,1] range if sigmoid

        // ArgMax for price level (e.g. if output has 10 logits for 10 levels)
        // For simplicity, assuming model just outputs a single level regression or we fix it to 0
        dec.price_level = 0;

        // Update internal state
        float executed_volume = remaining_volume_ * dec.trade_volume_ratio;
        remaining_volume_ -= executed_volume;
        time_elapsed_sec_ += dt_sec_; // Increment time

        return dec;
    }
};

// =================================================================================
// Public API
// =================================================================================

OrderExecutionPolicy::OrderExecutionPolicy(const ExecutionConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

OrderExecutionPolicy::~OrderExecutionPolicy() = default;
OrderExecutionPolicy::OrderExecutionPolicy(OrderExecutionPolicy&&) noexcept = default;
OrderExecution-Policy& OrderExecutionPolicy::operator=(OrderExecutionPolicy&&) noexcept = default;

void OrderExecutionPolicy::start_new_order(float total_volume, float total_time_sec) {
    if (pimpl_) pimpl_->reset(total_volume, total_time_sec);
}

ExecutionDecision OrderExecutionPolicy::get_action(const std::vector<float>& current_book_features) {
    if (!pimpl_) throw std::runtime_error("OrderExecutionPolicy is null.");

    // 1. Prepare Input
    pimpl_->prepare_input(current_book_features);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->decode_output();
}

} // namespace xinfer::zoo::hft