#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::retail {

    /**
     * @brief A single data point in time.
     */
    struct DailyRecord {
        float sales_volume; // The target variable
        float price;        // Exogenous variable
        float is_promotion; // 1.0 or 0.0
        float is_holiday;   // 1.0 or 0.0
    };

    /**
     * @brief Result of the forecast.
     */
    struct ForecastResult {
        // Predicted sales for the next N days
        std::vector<float> predicted_sales;

        // Confidence intervals (if model supports quantile regression)
        // std::vector<float> lower_bound;
        // std::vector<float> upper_bound;
    };

    struct DemandConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., lstm_demand.onnx)
        std::string model_path;

        // Sequence Settings
        int lookback_window = 30; // Past 30 days needed for context
        int forecast_horizon = 7; // Predict next 7 days

        // Normalization Stats (Critical for Regression)
        // Order: [Sales, Price, Promo, Holiday]
        std::vector<float> mean;
        std::vector<float> std;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DemandForecaster {
    public:
        explicit DemandForecaster(const DemandConfig& config);
        ~DemandForecaster();

        // Move semantics
        DemandForecaster(DemandForecaster&&) noexcept;
        DemandForecaster& operator=(DemandForecaster&&) noexcept;
        DemandForecaster(const DemandForecaster&) = delete;
        DemandForecaster& operator=(const DemandForecaster&) = delete;

        /**
         * @brief Add a new daily record to the history buffer.
         *
         * @param record Today's data.
         * @return True if sufficient history exists to forecast.
         */
        bool push_record(const DailyRecord& record);

        /**
         * @brief Predict future demand based on current history.
         *
         * @return ForecastResult containing denormalized sales numbers.
         */
        ForecastResult predict();

        /**
         * @brief Reset history (e.g. for a different SKU).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail