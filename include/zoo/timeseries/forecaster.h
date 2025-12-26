#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::timeseries {

    /**
     * @brief Result of a Forecast.
     */
    struct ForecastResult {
        // The predicted values for the future horizon.
        // Shape: [Horizon_Steps * Num_Output_Features]
        // If univariant: [t+1, t+2, t+3...]
        std::vector<float> values;
    };

    struct ForecasterConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., weather_lstm.onnx)
        std::string model_path;

        // Input Window (Lookback)
        // Number of past time steps the model needs.
        int input_window_size = 60;

        // Output Horizon (Forecast)
        // Number of future steps the model predicts.
        int output_horizon = 1;

        // Features
        int num_input_features = 1;
        int num_output_features = 1;

        // Normalization (Standard Scaling)
        // Crucial: The forecaster handles Denormalization of the output automatically.
        std::vector<float> mean; // Size = num_input_features
        std::vector<float> std;  // Size = num_input_features

        // If true, the output is denormalized using the same stats as input index 0.
        // (Assumes target variable is the first feature or univariate).
        bool auto_denormalize = true;
    };

    class Forecaster {
    public:
        explicit Forecaster(const ForecasterConfig& config);
        ~Forecaster();

        // Move semantics
        Forecaster(Forecaster&&) noexcept;
        Forecaster& operator=(Forecaster&&) noexcept;
        Forecaster(const Forecaster&) = delete;
        Forecaster& operator=(const Forecaster&) = delete;

        /**
         * @brief Push a new data point into the history buffer.
         *
         * @param features Vector of size 'num_input_features'.
         * @return True if sufficient history exists to forecast.
         */
        bool push(const std::vector<float>& features);

        /**
         * @brief Generate a forecast based on current history.
         *
         * @return ForecastResult containing the predicted future values (Denormalized).
         */
        ForecastResult predict();

        /**
         * @brief Clear history.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries