#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::timeseries {

    struct AnomalyResult {
        bool is_anomaly;
        float anomaly_score;
        std::vector<float> reconstruction_error;
    };

    struct AnomalyDetectorConfig {
        std::string engine_path;
        float anomaly_threshold = 0.1f;
        int sequence_length = 128;
    };

    class AnomalyDetector {
    public:
        explicit AnomalyDetector(const AnomalyDetectorConfig& config);
        ~AnomalyDetector();

        AnomalyDetector(const AnomalyDetector&) = delete;
        AnomalyDetector& operator=(const AnomalyDetector&) = delete;
        AnomalyDetector(AnomalyDetector&&) noexcept;
        AnomalyDetector& operator=(AnomalyDetector&&) noexcept;

        AnomalyResult predict(const std::vector<float>& time_series_window);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries

