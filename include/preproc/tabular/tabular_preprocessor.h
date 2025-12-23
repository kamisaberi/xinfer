#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::preproc {

    class ITabularPreprocessor {
    public:
        virtual ~ITabularPreprocessor() = default;

        /**
         * @brief Initialize with the dataset schema.
         * Contains the means/stds/categories learned during training.
         */
        virtual void init(const std::vector<tabular::ColumnSchema>& schema) = 0;

        /**
         * @brief Process a single log row into a Tensor.
         *
         * @param raw_row Vector of strings (e.g. ["TCP", "192.168.1.5", "80", "Error"])
         * @param dst Output tensor (Float32). Size = sum of encoded feature widths.
         */
        virtual void process(const tabular::TableRow& raw_row, core::Tensor& dst) = 0;

        /**
         * @brief Batch Process (Optimized for SIEM Throughput).
         */
        virtual void process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) = 0;
    };

}