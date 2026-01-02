#pragma once
#include <xinfer/telemetry/exporter.h>
#include <fstream>
#include <mutex>

namespace xinfer::telemetry {

    class FileExporter : public IMetricExporter {
    public:
        explicit FileExporter(const std::string& filepath);
        ~FileExporter() override;

        void export_metrics(const SystemMetrics& metrics) override;
        void export_inference(const InferenceMetrics& metrics) override;

    private:
        std::ofstream m_file;
        std::mutex m_mutex;
    };

}