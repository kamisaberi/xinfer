#include "log_encoder.h"
#include <xinfer/core/logging.h>
#include <cmath>
#include <cstring>
#include <algorithm>

namespace xinfer::preproc {

LogEncoder::LogEncoder() {}
LogEncoder::~LogEncoder() {}

void LogEncoder::init(const std::vector<tabular::ColumnSchema>& schema) {
    m_schema = schema;
    m_total_features = 0;

    // Calculate output vector size
    for (const auto& col : m_schema) {
        switch (col.type) {
            case tabular::ColumnType::NUMERICAL:
            case tabular::ColumnType::TIMESTAMP:
            case tabular::ColumnType::CATEGORICAL:
                m_total_features += 1;
                break;
            case tabular::ColumnType::IP_ADDRESS:
                m_total_features += 4; // Normalized octets
                break;
            case tabular::ColumnType::IGNORE:
                break;
        }
    }
}

// Optimized IP Parser: "192.168.1.1" -> {0.75, 0.66, 0.003, 0.003}
// Avoids memory allocation of stringstream
void LogEncoder::parse_ip_fast(const std::string& ip, float* out) {
    int octet = 0;
    int octet_idx = 0;

    for (char c : ip) {
        if (c >= '0' && c <= '9') {
            octet = octet * 10 + (c - '0');
        } else if (c == '.') {
            out[octet_idx++] = static_cast<float>(octet) / 255.0f;
            octet = 0;
            if (octet_idx >= 4) break;
        }
    }
    // Last octet
    if (octet_idx < 4) {
        out[octet_idx] = static_cast<float>(octet) / 255.0f;
    }
}

void LogEncoder::encode_row_to_buffer(const tabular::TableRow& row, float* buffer) {
    int buffer_idx = 0;

    // Safety check for row length
    size_t cols_to_process = std::min(row.size(), m_schema.size());

    for (size_t i = 0; i < cols_to_process; ++i) {
        const auto& col = m_schema[i];
        const std::string& val = row[i];

        if (col.type == tabular::ColumnType::IGNORE) continue;

        if (col.type == tabular::ColumnType::NUMERICAL ||
            col.type == tabular::ColumnType::TIMESTAMP) {

            float fval = 0.0f;
            try {
                fval = std::stof(val);
            } catch(...) { fval = 0.0f; } // Handle malformed logs gracefully

            if (col.encoding == tabular::EncodingType::STANDARD_SCALE) {
                // (x - u) / s
                buffer[buffer_idx++] = (fval - col.mean) / (col.std + 1e-6f);
            }
            else if (col.encoding == tabular::EncodingType::MIN_MAX_SCALE) {
                // (x - min) / (max - min)
                float denom = col.max - col.min;
                if (denom == 0) denom = 1.0f;
                buffer[buffer_idx++] = (fval - col.min) / denom;
            }
            else {
                buffer[buffer_idx++] = fval;
            }
        }
        else if (col.type == tabular::ColumnType::CATEGORICAL) {
            if (col.category_map.count(val)) {
                buffer[buffer_idx++] = col.category_map.at(val);
            } else {
                buffer[buffer_idx++] = col.unknown_value;
            }
        }
        else if (col.type == tabular::ColumnType::IP_ADDRESS) {
            parse_ip_fast(val, &buffer[buffer_idx]);
            buffer_idx += 4;
        }
    }
}

void LogEncoder::process(const tabular::TableRow& raw_row, core::Tensor& dst) {
    if (dst.empty() || dst.size() != m_total_features) {
        dst.resize({1, (int64_t)m_total_features}, core::DataType::kFLOAT);
    }

    encode_row_to_buffer(raw_row, static_cast<float*>(dst.data()));
}

void LogEncoder::process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) {
    size_t batch_size = rows.size();

    // Resize Output Tensor [Batch, Features]
    dst.resize({(int64_t)batch_size, (int64_t)m_total_features}, core::DataType::kFLOAT);

    float* raw_ptr = static_cast<float*>(dst.data());

    // Loop could be parallelized with OpenMP for CPU speedup on servers
    // #pragma omp parallel for
    for (size_t i = 0; i < batch_size; ++i) {
        float* row_ptr = raw_ptr + (i * m_total_features);
        encode_row_to_buffer(rows[i], row_ptr);
    }
}

} // namespace xinfer::preproc