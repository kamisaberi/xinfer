#include <xinfer/telemetry/exporters/file_exporter.h>
#include <xinfer/core/logging.h>
#include "json.hpp" // nlohmann/json

using json = nlohmann::json;

namespace xinfer::telemetry {

    FileExporter::FileExporter(const std::string& filepath) {
        // Open in append mode
        m_file.open(filepath, std::ios::app);
        if (!m_file.is_open()) {
            XINFER_LOG_ERROR("Telemetry: Failed to open log file: " + filepath);
        }
    }

    FileExporter::~FileExporter() {
        if (m_file.is_open()) {
            m_file.close();
        }
    }

    void FileExporter::export_metrics(const SystemMetrics& metrics) {
        if (!m_file.is_open()) return;

        json j;
        j["type"] = "system";
        j["timestamp"] = metrics.timestamp_ms;
        j["cpu_percent"] = metrics.cpu_usage_percent;
        j["ram_mb"] = metrics.ram_usage_mb;
        j["gpu_temp"] = metrics.gpu_temp_c;

        std::lock_guard<std::mutex> lock(m_mutex);
        m_file << j.dump() << std::endl;
    }

    void FileExporter::export_inference(const InferenceMetrics& metrics) {
        if (!m_file.is_open()) return;

        json j;
        j["type"] = "inference";
        j["model"] = metrics.model_name;
        j["fps"] = metrics.current_fps;
        j["latency_avg"] = metrics.avg_latency_ms;
        j["latency_p99"] = metrics.p99_latency_ms;

        std::lock_guard<std::mutex> lock(m_mutex);
        m_file << j.dump() << std::endl;
    }

} // namespace xinfer::telemetry