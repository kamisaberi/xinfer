#include <iostream>
#include <thread>
#include <chrono>

// xInfer Telemetry
#include <xinfer/telemetry/monitor.h>
#include <xinfer/core/logging.h>

using namespace xinfer::telemetry;

int main() {
    std::cout << "--- xInfer System Health Monitor ---" << std::endl;

    // 1. Initialize Monitor
    Monitor monitor;

    // 2. Start Background Polling (1000ms interval)
    // This spawns a thread that reads /proc/stat, /sys/class/thermal, etc.
    monitor.start(1000);

    std::cout << "Monitoring started. Press Ctrl+C to stop." << std::endl;

    // 3. Simulate an Application Loop
    for (int i = 0; i < 10; ++i) {
        // Do some heavy work here...
        std::this_thread::sleep_for(std::chrono::seconds(2));

        // 4. Query Metrics
        // This is non-blocking (reads from cached state protected by mutex)
        SystemMetrics m = monitor.get_metrics();

        std::cout << "[Tick " << i << "] "
                  << "CPU: " << m.cpu_usage_percent << "% | "
                  << "RAM: " << m.ram_usage_mb << "/" << m.ram_total_mb << " MB | "
                  << "Temp: " << m.gpu_temp_c << "C" << std::endl;

        // 5. Thermal Throttling Logic
        if (m.gpu_temp_c > 80.0) {
            std::cerr << "CRITICAL WARNING: GPU Overheating! Throttling inference..." << std::endl;
            // Logic to sleep longer or skip frames would go here
        }

        // 6. Export for Logging
        // Returns a JSON string suitable for REST APIs
        // std::string json = monitor.export_json();
        // save_to_log(json);
    }

    // 7. Cleanup
    monitor.stop();
    return 0;
}