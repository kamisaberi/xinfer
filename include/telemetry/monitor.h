#pragma once

#include "types.h"
#include <memory>
#include <string>

namespace xinfer::telemetry {

    /**
     * @brief System Health Monitor.
     *
     * Runs a lightweight background thread to poll /proc (Linux) or system APIs.
     * Essential for preventing thermal throttling on Jetson/Rockchip devices.
     */
    class Monitor {
    public:
        Monitor();
        ~Monitor();

        // Starts the background polling thread
        void start(int interval_ms = 1000);
        void stop();

        // Get the latest snapshot
        SystemMetrics get_metrics() const;

        // Export current state to JSON string (for the Serving API)
        std::string export_json() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::telemetry