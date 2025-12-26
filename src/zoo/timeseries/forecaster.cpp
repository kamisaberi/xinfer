#include <xinfer/zoo/timeseries/forecaster.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// No heavy preproc factory needed; time-series math handled internally.

#include <iostream>
#include <deque>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::timeseries {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Forecaster::Impl {
    ForecasterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Sliding Window Buffer (Stores Raw Data)
    std::deque<std::vector<float>> window_buffer_;

    Impl(const ForecasterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Forecaster: Failed to load model " + config_.model_path);
        }

        // Validate Norm Params
        if (!config_.mean.empty() && config_.mean.size() != config_.num_input_features) {
            XINFER_LOG_WARN("Forecaster: Mean vector size mismatch. Normalization disabled.");
            // Logic to disable norm could be added here
        }

        // 2. Allocate Input Tensor
        // Shape: [1, Window, Features] (Standard LSTM/Transformer input)
        input_tensor.resize({1, (int64_t)config_.input_window_size, (int64_t)config_.num_input_features}, core::DataType::kFLOAT);
    }

    void add_point(const std::vector<float>& features) {
        if (features.size() != config_.num_input_features) {
            XINFER_LOG_ERROR("Input feature size mismatch.");
            return;
        }
        window_buffer_.push_back(features);
        if (window_buffer_.size() > config_.input_window_size) {
            window_buffer_.pop_front();
        }
    }

    void prepare_input() {
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        bool use_norm = !config_.mean.empty();

        // Iterate Time
        for (const auto& step : window_buffer_) {
            // Iterate Features
            for (int f = 0; f < config_.num_input_features; ++f) {
                float val = step[f];
                // Normalize: (x - u) / s
                if (use_norm) {
                    float s = config_.std[f];
                    if (std::abs(s) < 1e-6) s = 1.0f;
                    val = (val - config_.mean[f]) / s;
                }
                ptr[idx++] = val;
            }
        }
    }

    ForecastResult postprocess() {
        ForecastResult res;

        const float* out_ptr = static_cast<const float*>(output_tensor.data());
        size_t count = output_tensor.size(); // Total output elements (Horizon * OutFeatures)

        res.values.resize(count);
        bool use_denorm = config_.auto_denormalize && !config_.mean.empty();

        // 1. Copy & Denormalize
        for (size_t i = 0; i < count; ++i) {
            float val = out_ptr[i];

            if (use_denorm) {
                // Determine which feature index this value corresponds to.
                // Assuming output layout is [Horizon, OutFeatures] or [1, Horizon, OutFeatures]

                int feature_idx = i % config_.num_output_features;

                // If output features match input features (multivariate forecast), map 1:1.
                // If output is univariate (1 feature) but input was multi, assume target was index 0.
                if (feature_idx < config_.std.size()) {
                    float s = config_.std[feature_idx];
                    float m = config_.mean[feature_idx];
                    // Denormalize: y = y' * s + u
                    val = (val * s) + m;
                }
            }
            res.values[i] = val;
        }

        return res;
    }
};

// =================================================================================
// Public API
// =================================================================================

Forecaster::Forecaster(const ForecasterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Forecaster::~Forecaster() = default;
Forecaster::Forecaster(Forecaster&&) noexcept = default;
Forecaster& Forecaster::operator=(Forecaster&&) noexcept = default;

void Forecaster::reset() {
    if (pimpl_) pimpl_->window_buffer_.clear();
}

bool Forecaster::push(const std::vector<float>& features) {
    if (!pimpl_) return false;
    pimpl_->add_point(features);
    return (pimpl_->window_buffer_.size() >= pimpl_->config_.input_window_size);
}

ForecastResult Forecaster::predict() {
    if (!pimpl_) throw std::runtime_error("Forecaster is null.");

    if (pimpl_->window_buffer_.size() < pimpl_->config_.input_window_size) {
        XINFER_LOG_WARN("Forecaster: Insufficient history. Prediction may be garbage (Zero padded).");
        // In robust app, return empty result or handle padding explicitly
    }

    // 1. Prepare
    pimpl_->prepare_input();

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Denormalize)
    return pimpl_->postprocess();
}

} // namespace xinfer::zoo::timeseries