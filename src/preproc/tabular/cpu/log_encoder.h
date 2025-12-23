#pragma once

#include <xinfer/preproc/tabular/tabular_preprocessor.h>

namespace xinfer::preproc {

    class LogEncoder : public ITabularPreprocessor {
    public:
        LogEncoder();
        ~LogEncoder() override;

        void init(const std::vector<tabular::ColumnSchema>& schema) override;

        void process(const tabular::TableRow& raw_row, core::Tensor& dst) override;
        void process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) override;
        size_t get_output_width() const override { return m_total_features; }

    private:
        std::vector<tabular::ColumnSchema> m_schema;
        size_t m_total_features = 0;

        // --- Optimized Helpers ---
        // Fast string-to-int/float conversions
        void encode_row_to_buffer(const tabular::TableRow& row, float* buffer);

        // Optimized IP Parser (avoids sscanf/stringstream overhead)
        void parse_ip_fast(const std::string& ip, float* out);
    };

} // namespace xinfer::preproc