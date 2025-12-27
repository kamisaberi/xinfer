#include <xinfer/zoo/fashion/trend_forecaster.h>
#include <xinfer/core/logging.h>

// --- We reuse the generic Time Series Forecaster ---
#include <xinfer/zoo/timeseries/forecaster.h>

#include <iostream>
#include <vector>

namespace xinfer::zoo::fashion {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TrendForecaster::Impl {
    ForecasterConfig config_;

    // The generic Time Series Forecaster does all the heavy lifting
    std::unique_p<timeseries::Forecaster> ts_engine_;

    // Feature order mapping
    // 0: Sales, 1: Price, 2: Mentions, 3: Volume, 4: Color, 5: Category
    const int NUM_FEATURES = 6;

    Impl(const ForecasterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the underlying time series forecaster
        timeseries::ForecasterConfig ts_cfg;
        ts_cfg.target = config_.target;
        ts_cfg.model_path = config_.model_path;
        ts_cfg.input_window_size = config_.lookback_window;
        ts_cfg.output_horizon = config_.forecast_horizon;
        ts_cfg.num_input_features = NUM_FEATURES;
        ts_cfg.num_output_features = 1; // Predict Sales only
        ts_cfg.mean = config_.mean;
        ts_cfg.std = config_.std;

        ts_engine_ = std::make_unique<timeseries::Forecaster>(ts_cfg);
    }

    // --- Core Logic: Domain Struct -> Generic Vector ---
    std::vector<float> map_features(const TrendDataPoint& data) {
        std::vector<float> features(NUM_FEATURES, 0.0f);

        features[0] = data.sales_volume;
        features[1] = data.price;
        features[2] = data.social_media_mentions;
        features[3] = data.search_volume;

        // Categorical Encoding
        if (config_.color_map.count(data.color)) {
            features[4] = config_.color_map.at(data.color);
        }
        if (config_.category_map.count(data.category)) {
            features[5] = config_.category_map.at(data.category);
        }

        return features;
    }
};

// =================================================================================
// Public API
// =================================================================================

TrendForecaster::TrendForecaster(const ForecasterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TrendForecaster::~TrendForecaster() = default;
TrendForecaster::TrendForecaster(TrendForecaster&&) noexcept = default;
TrendForecaster& TrendForecaster::operator=(TrendForecaster&&) noexcept = default;

void TrendForecaster::reset() {
    if (pimpl_ && pimpl_->ts_engine_) pimpl_->ts_engine_->reset();
}

bool TrendForecaster::push_data(const TrendDataPoint& data) {
    if (!pimpl_ || !pimpl_->ts_engine_) return false;

    auto features = pimpl_->map_features(data);
    return pimpl_->ts_engine_->push(features);
}

TrendForecast TrendForecaster::predict() {
    if (!pimpl_ || !pimpl_->ts_engine_) throw std::runtime_error("TrendForecaster is null.");

    // 1. Run generic forecast
    auto raw_forecast = pimpl_->ts_engine_->predict();

    // 2. Add domain-specific logic
    TrendForecast result;
    result.future_sales = raw_forecast.values;

    // Simple trend analysis: Is the end of the forecast higher than the start?
    if (result.future_sales.size() > 1) {
        result.is_trending_up = result.future_sales.back() > result.future_sales.front();
    } else {
        result.is_trending_up = false;
    }

    return result;
}

} // namespace xinfer::zoo::fashion