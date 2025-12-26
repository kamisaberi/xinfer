#include <xinfer/zoo/retail/demand_forecaster.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Standard preproc not used; custom time-series normalization logic implemented below.

#include <iostream>
#include <deque>
#include <vector>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::retail {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DemandForecaster::Impl {
    DemandConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // History Buffer (Raw values)
    // We store vectors of size 4: [Sales, Price, Promo, Holiday]
    std::deque<std::vector<float>> history_;

    // Constants
    const int NUM_FEATURES = 4;

    Impl(const DemandConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DemandForecaster: Failed to load model " + config_.model_path);
        }

        // Validate Normalization Config
        if (config_.mean.size() != NUM_FEATURES || config_.std.size() != NUM_FEATURES) {
            XINFER_LOG_WARN("Normalization stats size mismatch. Defaulting to raw values.");
        }

        // 2. Pre-allocate Input Tensor
        // Shape: [1, Lookback, Features]
        input_tensor.resize({1, (int64_t)config_.lookback_window, (int64_t)NUM_FEATURES}, core::DataType::kFLOAT);
    }

    void add_to_history(const DailyRecord& r) {
        std::vector<float> vec = {r.sales_volume, r.price, r.is_promotion, r.is_holiday};
        history_.push_back(vec);

        if (history_.size() > config_.lookback_window) {
            history_.pop_front();
        }
    }

    void prepare_tensor() {
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        bool do_norm = (config_.mean.size() == NUM_FEATURES);

        // Flatten history into tensor
        for (const auto& step : history_) {
            for (int f = 0; f < NUM_FEATURES; ++f) {
                float val = step[f];
                if (do_norm) {
                    // (x - u) / s
                    val = (val - config_.mean[f]) / (config_.std[f] + 1e-6f);
                }
                ptr[idx++] = val;
            }
        }

        // Pad with zeros if history is not full yet (Cold Start)
        // (Though push_record returns false in that case, good to be safe)
        int remaining = (config_.lookback_window * NUM_FEATURES) - idx;
        if (remaining > 0) {
             // In time series, usually padding goes at the START, but here we just zero fill.
             // Ideally, don't predict until buffer full.
        }
    }

    ForecastResult process_output() {
        ForecastResult res;

        // Output shape usually: [1, Horizon, 1] or [1, Horizon]
        const float* out_ptr = static_cast<const float*>(output_tensor.data());
        bool do_denorm = (config_.mean.size() == NUM_FEATURES);

        for (int i = 0; i < config_.forecast_horizon; ++i) {
            float val = out_ptr[i];

            // Denormalize
            // We assume the model predicts 'Sales' which is index 0 in our feature list
            if (do_denorm) {
                // y = y_norm * std[0] + mean[0]
                val = val * config_.std[0] + config_.mean[0];
            }

            // Sales can't be negative
            val = std::max(0.0f, val);

            res.predicted_sales.push_back(val);
        }

        return res;
    }
};

// =================================================================================
// Public API
// =================================================================================

DemandForecaster::DemandForecaster(const DemandConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DemandForecaster::~DemandForecaster() = default;
DemandForecaster::DemandForecaster(DemandForecaster&&) noexcept = default;
DemandForecaster& DemandForecaster::operator=(DemandForecaster&&) noexcept = default;

void DemandForecaster::reset() {
    if (pimpl_) pimpl_->history_.clear();
}

bool DemandForecaster::push_record(const DailyRecord& record) {
    if (!pimpl_) throw std::runtime_error("DemandForecaster is null.");

    pimpl_->add_to_history(record);
    return (pimpl_->history_.size() >= pimpl_->config_.lookback_window);
}

ForecastResult DemandForecaster::predict() {
    if (!pimpl_) throw std::runtime_error("DemandForecaster is null.");

    if (pimpl_->history_.size() < pimpl_->config_.lookback_window) {
        XINFER_LOG_WARN("DemandForecaster: Insufficient history. Prediction accuracy will be low.");
    }

    // 1. Prepare Input
    pimpl_->prepare_tensor();

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->process_output();
}

} // namespace xinfer::zoo::retail