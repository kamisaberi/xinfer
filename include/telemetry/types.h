#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace xinfer::telemetry {

    /**
     * @brief Snapshot of hardware health.
     */
    struct SystemMetrics {
        double cpu_usage_percent;  // 0.0 - 100.0
        double ram_usage_mb;       // Used memory in MB
        double ram_total_mb;       // Total memory
        double gpu_usage_percent;  // 0.0 - 100.0 (if available)
        double gpu_temp_c;         // Temperature in Celsius
        uint64_t timestamp_ms;
    };

    /**
     * @brief Performance stats for a specific model.
     */
    struct InferenceMetrics {
        std::string model_name;
        double current_fps;
        double avg_latency_ms;
        double p99_latency_ms;
        uint64_t total_requests;
    };

    /**
     * @brief Result of Data Drift analysis.
     */
    struct DriftResult {
        bool has_drift;           // True if Z-Score > threshold
        float drift_score;        // Statistical deviation amount
        std::string feature_name; // Which feature drifted (or "global")
    };

} // namespace xinfer::telemetry