#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::timeseries {

    /**
     * @brief Result of Time Series Anomaly Analysis.
     */
    struct TSAnomalyResult {
        bool is_anomaly;        // True if score > threshold
        float anomaly_score;    // Reconstruction Error (MSE/MAE)

        // The values that were reconstructed/predicted by the model
        std::vector<float> reconstruction;
    };

    struct TSAnomalyConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., lstm_ae.onnx, dense_ae.rknn)
        std::string model_path;

        // Window Size (Sequence Length)
        // The model expects inputs of shape [1, window_size, features]
        int window_size = 30;

        // Number of features per time step (e.g. 1 for just temperature, 3 for accel X/Y/Z)
        int num_features = 1;

        // Normalization (Standard Scaling: (x - mean) / std)
        // Critical for Time Series models.
        std::vector<float> mean; // Size must equal num_features
        std::vector<float> std;  // Size must equal num_features

        // Sensitivity
        float threshold = 0.5f;
    };

    class AnomalyDetector {
    public:
        explicit AnomalyDetector(const TSAnomalyConfig& config);
        ~AnomalyDetector();

        // Move semantics
        AnomalyDetector(AnomalyDetector&&) noexcept;
        AnomalyDetector& operator=(AnomalyDetector&&) noexcept;
        AnomalyDetector(const AnomalyDetector&) = delete;
        AnomalyDetector& operator=(const AnomalyDetector&) = delete;

        /**
         * @brief Push a new data point into the sliding window.
         *
         * @param features Vector of size 'num_features'.
         * @return True if window is full and ready for detection, False otherwise.
         */
        bool push(const std::vector<float>& features);

        /**
         * @brief Run anomaly detection on the current window.
         *
         * Logic:
         * 1. Normalize current window.
         * 2. Inference (Reconstruction).
         * 3. Calculate Error (MSE) between Input Window and Output.
         */
        TSAnomalyResult detect();

        /**
         * @brief Clear the internal sliding window history.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries