#include <xinfer/telemetry/exporters/metric_exporter.h>
#include <xinfer/core/logging.h>

#include "httplib.h" // third_party/httplib.h
#include <sstream>

namespace xinfer::telemetry {

struct HttpMetricExporter::Impl {
    std::string host;
    int port;
    std::string endpoint;
    httplib::Client cli;

    Impl(const std::string& h, int p, const std::string& e)
        : host(h), port(p), endpoint(e), cli(h, p)
    {
        cli.set_connection_timeout(0, 500000); // 0.5 sec timeout
    }

    // Convert metrics to Prometheus Text Format
    // metric_name{label="value"} value
    void send_prometheus(const std::string& payload) {
        // Fire and forget (don't block inference for metrics)
        auto res = cli.Post(endpoint.c_str(), payload, "text/plain");
        if (!res || res->status != 200) {
            // XINFER_LOG_WARN("Failed to push metrics");
        }
    }
};

HttpMetricExporter::HttpMetricExporter(const std::string& host, int port, const std::string& endpoint)
    : pimpl_(std::make_unique<Impl>(host, port, endpoint)) {}

HttpMetricExporter::~HttpMetricExporter() = default;

void HttpMetricExporter::export_metrics(const SystemMetrics& metrics) {
    std::stringstream ss;

    // CPU
    ss << "xinfer_cpu_usage " << metrics.cpu_usage_percent << "\n";

    // RAM
    ss << "xinfer_ram_usage_mb " << metrics.ram_usage_mb << "\n";

    // GPU
    if (metrics.gpu_temp_c > 0) {
        ss << "xinfer_gpu_temp_c " << metrics.gpu_temp_c << "\n";
    }

    pimpl_->send_prometheus(ss.str());
}

void HttpMetricExporter::export_inference(const InferenceMetrics& metrics) {
    std::stringstream ss;

    // FPS
    ss << "xinfer_inference_fps{model=\"" << metrics.model_name << "\"} " << metrics.current_fps << "\n";

    // Latency
    ss << "xinfer_latency_ms{model=\"" << metrics.model_name << "\",type=\"avg\"} " << metrics.avg_latency_ms << "\n";
    ss << "xinfer_latency_ms{model=\"" << metrics.model_name << "\",type=\"p99\"} " << metrics.p99_latency_ms << "\n";

    pimpl_->send_prometheus(ss.str());
}

} // namespace xinfer::telemetry