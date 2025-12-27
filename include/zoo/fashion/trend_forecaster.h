#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::fashion {

    /**
     * @brief A single data point (e.g., daily metrics).
     */
    struct TrendDataPoint {
        // Target Variable
        float sales_volume;

        // Exogenous Variables (External Factors)
        float price;
        float social_media_mentions;
        float search_volume;

        // Categorical
        std::string color;
        std::string category; // "Shoes", "Dresses"
    };

    /**
     * @brief Result of the forecast.
     */
    struct TrendForecast {
        // Predicted sales for the next N time steps
        std::vector<float> future_sales;

        // Indicates if the trend is increasing or decreasing
        bool is_trending_up;
    };

    struct ForecasterConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., fashion_trend_lstm.onnx)
        std::string model_path;

        // --- Model Specs ---
        int lookback_window = 90; // Use past 90 days
        int forecast_horizon = 30;  // Predict next 30 days

        // --- Normalization & Encoding ---
        // For Numerical features (Sales, Price, Mentions, Volume)
        std::vector<float> mean;
        std::vector<float> std;

        // For Categorical features (Color, Category)
        // Map: "red" -> 0, "blue" -> 1, etc.
        std::map<std::string, float> color_map;
        std::map<std::string, float> category_map;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TrendForecaster {
    public:
        explicit TrendForecaster(const ForecasterConfig& config);
        ~TrendForecaster();

        // Move semantics
        TrendForecaster(TrendForecaster&&) noexcept;
        TrendForecaster& operator=(TrendForecaster&&) noexcept;
        TrendForecaster(const TrendForecaster&) = delete;
        TrendForecaster& operator=(const TrendForecaster&) = delete;

        /**
         * @brief Add a new data point to the history.
         */
        bool push_data(const TrendDataPoint& data);

        /**
         * @brief Predict the next trend based on current history.
         */
        TrendForecast predict();

        /**
         * @brief Reset history.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::fashion