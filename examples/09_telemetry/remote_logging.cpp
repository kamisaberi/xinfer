#include <iostream>
#include <thread>
#include <vector>

// We include the internal exporter headers directly for this example.
// In a real install, these would be under <xinfer/telemetry/exporters/...>
#include "../src/telemetry/exporters/metric_exporter.cpp" // Direct include for demo simplicity if libs not installed
// OR if installed: #include <xinfer/telemetry/exporters/metric_exporter.h>

using namespace xinfer::telemetry;

int main() {
    std::cout << "--- xInfer Remote Metrics Pusher ---" << std::endl;

    // 1. Setup Exporter
    // Push to a local Prometheus Pushgateway
    std::string host = "localhost";
    int port = 9091;
    std::string endpoint = "/metrics/job/xinfer_vision";

    // Note: HttpMetricExporter implementation needs to be linked
    // Assuming we are linking against xinfer_telemetry library
    HttpMetricExporter exporter(host, port, endpoint);

    std::cout << "Pushing metrics to http://" << host << ":" << port << endpoint << std::endl;

    // 2. Simulate Inference Loop
    int frame_count = 0;

    while (frame_count < 10) {
        // Simulate Inference
        auto start = std::chrono::high_resolution_clock::now();
        std::this_thread::sleep_for(std::chrono::milliseconds(33)); // ~30 FPS
        auto end = std::chrono::high_resolution_clock::now();

        double latency = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // 3. Prepare Metrics Packet
        InferenceMetrics metrics;
        metrics.model_name = "yolov8_traffic_cam_04";
        metrics.current_fps = 1000.0 / latency;
        metrics.avg_latency_ms = latency;
        metrics.p99_latency_ms = latency + 2.0; // Simulated jitter
        metrics.total_requests = ++frame_count;

        // 4. Export
        std::cout << "Pushing frame " << frame_count << " stats..." << std::endl;
        exporter.export_inference(metrics);
    }

    return 0;
}