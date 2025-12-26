#include <xinfer/zoo/timeseries/anomaly_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Note: Time Series preprocessing is simple math, handled internally here
// rather than using the heavy image/audio preprocessors.

#include <iostream>
#include <deque>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace xinfer::zoo::timeseries {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct AnomalyDetector::Impl {
    TSAnomalyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Sliding Window Buffer
    // Stores raw (un-normalized) data
    std::deque<std::vector<float>> window_buffer_;

    Impl(const TSAnomalyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TSAnomalyDetector: Failed to load model " + config_.model_path);
        }

        // Validate Config
        if (config_.mean.size() != config_.num_features || config_.std.size() != config_.num_features) {
            XINFER_LOG_WARN("Normalization params size mismatch. Defaulting to Mean=0, Std=1.");
            // Fix config locally if needed, or rely on push() to handle raw data if empty
        }

        // Pre-allocate Input Tensor [1, WindowSize, NumFeatures]
        // Note: Layout depends on model (LSTM usually takes [Batch, Time, Feat])
        input_tensor.resize({1, (int64_t)config_.window_size, (int64_t)config_.num_features}, core::DataType::kFLOAT);
    }

    void add_point(const std::vector<float>& features) {
        if (features.size() != config_.num_features) {
            XINFER_LOG_ERROR("Input feature size mismatch.");
            return;
        }

        window_buffer_.push_back(features);
        if (window_buffer_.size() > config_.window_size) {
            window_buffer_.pop_front();
        }
    }

    // Prepare tensor: Flatten window buffer and Normalize
    void prepare_input() {
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;

        bool use_norm = (!config_.mean.empty() && !config_.std.empty());

        for (const auto& step : window_buffer_) {
            for (int f = 0; f < config_.num_features; ++f) {
                float val = step[f];
                if (use_norm) {
                    val = (val - config_.mean[f]) / (config_.std[f] + 1e-6f);
                }
                ptr[idx++] = val;
            }
        }
    }

    // Post-process: Calculate MSE
    TSAnomalyResult compute_result() {
        TSAnomalyResult res;
        res.is_anomaly = false;
        res.anomaly_score = 0.0f;

        // Get Input Data (Normalized)
        const float* in_ptr = static_cast<const float*>(input_tensor.data());

        // Get Output Data (Reconstruction)
        const float* out_ptr = static_cast<const float*>(output_tensor.data());

        // Store reconstruction for user (Denormalized? Optional. Here returning raw model output)
        size_t total_elements = config_.window_size * config_.num_features;
        res.reconstruction.assign(out_ptr, out_ptr + total_elements);

        // Calculate MSE (Mean Squared Error)
        float sum_sq_err = 0.0f;
        for (size_t i = 0; i < total_elements; ++i) {
            float diff = in_ptr[i] - out_ptr[i];
            sum_sq_err += (diff * diff);
        }
        res.anomaly_score = sum_sq_err / total_elements;

        // Thresholding
        if (res.anomaly_score > config_.threshold) {
            res.is_anomaly = true;
        }

        return res;
    }
};

// =================================================================================
// Public API
// =================================================================================

AnomalyDetector::AnomalyDetector(const TSAnomalyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

AnomalyDetector::~AnomalyDetector() = default;
AnomalyDetector::AnomalyDetector(AnomalyDetector&&) noexcept = default;
AnomalyDetector& AnomalyDetector::operator=(AnomalyDetector&&) noexcept = default;

void AnomalyDetector::reset() {
    if (pimpl_) pimpl_->window_buffer_.clear();
}

bool AnomalyDetector::push(const std::vector<float>& features) {
    if (!pimpl_) return false;
    pimpl_->add_point(features);
    return (pimpl_->window_buffer_.size() >= pimpl_->config_.window_size);
}

TSAnomalyResult AnomalyDetector::detect() {
    if (!pimpl_) throw std::runtime_error("TSAnomalyDetector is null.");

    // Check if we have enough data
    if (pimpl_->window_buffer_.size() < pimpl_->config_.window_size) {
        XINFER_LOG_WARN("Not enough data points for full window inference. Padding with zeros might occur.");
        // In this implementation, we just proceed (buffer might be partial),
        // but typically you wait until push returns true.
    }

    // 1. Prepare Input (Normalize & Flatten)
    pimpl_->prepare_input();

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (MSE)
    return pimpl_->compute_result();
}

} // namespace xinfer::zoo::timeseries