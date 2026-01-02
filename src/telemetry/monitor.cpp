#include <xinfer/telemetry/monitor.h>
#include <xinfer/core/logging.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <atomic>
#include <chrono>
#include <mutex>
#include <vector>
#include <cstring>

namespace xinfer::telemetry {

struct Monitor::Impl {
    SystemMetrics current_metrics_;
    std::atomic<bool> running_{false};
    std::thread worker_;
    mutable std::mutex mutex_;
    int interval_ms_ = 1000;

    // CPU Calculation State
    unsigned long long prev_user_ = 0, prev_nice_ = 0, prev_system_ = 0, prev_idle_ = 0;

    Impl() {
        // Init zero
        std::memset(&current_metrics_, 0, sizeof(SystemMetrics));
    }

    void poll_system() {
        while (running_) {
            SystemMetrics m;
            m.timestamp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();

            // 1. Read RAM (Linux /proc/meminfo)
            std::ifstream mem_file("/proc/meminfo");
            if (mem_file.is_open()) {
                std::string line, key;
                long value;
                long total = 0, available = 0;
                while (std::getline(mem_file, line)) {
                    std::stringstream ss(line);
                    ss >> key >> value;
                    if (key == "MemTotal:") total = value;
                    else if (key == "MemAvailable:") available = value;
                }
                if (total > 0) {
                    m.ram_total_mb = total / 1024.0;
                    m.ram_usage_mb = (total - available) / 1024.0;
                }
            }

            // 2. Read CPU Load (/proc/stat)
            std::ifstream cpu_file("/proc/stat");
            if (cpu_file.is_open()) {
                std::string line, cpu_label;
                unsigned long long user, nice, system, idle;
                std::getline(cpu_file, line);
                std::stringstream ss(line);
                ss >> cpu_label >> user >> nice >> system >> idle;

                unsigned long long total = user + nice + system + idle;
                unsigned long long total_diff = total - (prev_user_ + prev_nice_ + prev_system_ + prev_idle_);
                unsigned long long idle_diff = idle - prev_idle_;

                if (total_diff > 0) {
                    m.cpu_usage_percent = 100.0 * (1.0 - ((double)idle_diff / total_diff));
                }

                prev_user_ = user; prev_nice_ = nice; prev_system_ = system; prev_idle_ = idle;
            }

            // 3. Read Thermal (/sys/class/thermal/thermal_zone0/temp)
            // Works on Jetson and most embedded Linux
            std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
            if (temp_file.is_open()) {
                long temp_milli;
                temp_file >> temp_milli;
                m.gpu_temp_c = temp_milli / 1000.0; // Often system temp acts as proxy
            }

            // Lock and update
            {
                std::lock_guard<std::mutex> lock(mutex_);
                current_metrics_ = m;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
        }
    }
};

Monitor::Monitor() : pimpl_(std::make_unique<Impl>()) {}
Monitor::~Monitor() { stop(); }

void Monitor::start(int interval_ms) {
    if (pimpl_->running_) return;
    pimpl_->interval_ms_ = interval_ms;
    pimpl_->running_ = true;
    pimpl_->worker_ = std::thread(&Impl::poll_system, pimpl_.get());
    XINFER_LOG_INFO("Telemetry Monitor started.");
}

void Monitor::stop() {
    pimpl_->running_ = false;
    if (pimpl_->worker_.joinable()) {
        pimpl_->worker_.join();
    }
}

SystemMetrics Monitor::get_metrics() const {
    std::lock_guard<std::mutex> lock(pimpl_->mutex_);
    return pimpl_->current_metrics_;
}

std::string Monitor::export_json() const {
    auto m = get_metrics();
    std::stringstream ss;
    ss << "{";
    ss << "\"cpu_usage\": " << m.cpu_usage_percent << ",";
    ss << "\"ram_usage_mb\": " << m.ram_usage_mb << ",";
    ss << "\"temp_c\": " << m.gpu_temp_c << ",";
    ss << "\"timestamp\": " << m.timestamp_ms;
    ss << "}";
    return ss.str();
}

} // namespace xinfer::telemetry