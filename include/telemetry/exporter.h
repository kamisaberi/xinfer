#pragma once
#include <xinfer/telemetry/types.h>

namespace xinfer::telemetry {

    class IMetricExporter {
    public:
        virtual ~IMetricExporter() = default;

        /**
         * @brief Export system metrics to a destination.
         */
        virtual void export_metrics(const SystemMetrics& metrics) = 0;

        /**
         * @brief Export inference specific metrics.
         */
        virtual void export_inference(const InferenceMetrics& metrics) = 0;
    };

}