#pragma once
#include <xinfer/telemetry/exporter.h>
#include <memory>
#include <string>

namespace xinfer::telemetry {

    class HttpMetricExporter : public IMetricExporter {
    public:
        /**
         * @param host Target host (e.g. "localhost" or "monitoring.internal")
         * @param port Target port (e.g. 9091)
         * @param endpoint URL path (e.g. "/metrics/job/xinfer")
         */
        HttpMetricExporter(const std::string& host, int port, const std::string& endpoint);
        ~HttpMetricExporter() override;

        void export_metrics(const SystemMetrics& metrics) override;
        void export_inference(const InferenceMetrics& metrics) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

}