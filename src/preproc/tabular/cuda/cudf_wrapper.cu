#include "cudf_wrapper.h"
#include <xinfer/core/logging.h>

#ifdef XINFER_HAS_CUDA

// --- cuDF Headers ---
#include <cudf/column/column_factories.hpp>
#include <cudf/io/csv.hpp> // For reading large batches
#include <cudf/strings/strings_column.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/strings/regex/findall.hpp>
#include <cudf/strings/split/split.hpp>
#include <cudf/transform.hpp>
#include <cudf/types.hpp>
#include <cudf/unary.hpp>
#include <cudf/replace.hpp>
#include <cudf/ast/ast.hpp>
#include <cudf/copying.hpp>
#include <cudf/concatenate.hpp>

// --- RMM Headers (CUDA Memory Management) ---
#include <rmm/mr/device/per_device_resource.hpp>
#include <rmm/cuda_stream_view.hpp>

namespace xinfer::preproc {

// =================================================================================
// CUDA cuDF Implementation
// =================================================================================

CudfTabularPreprocessor::CudfTabularPreprocessor() {
    // Initialize RMM (RAPIDS Memory Manager)
    rmm::mr::set_current_device_resource(rmm::mr::get_per_device_resource(0));
    cudaStreamCreate(&m_cuda_stream);
}

CudfTabularPreprocessor::~CudfTabularPreprocessor() {
    if (m_cuda_stream) cudaStreamDestroy(m_cuda_stream);
}

void CudfTabularPreprocessor::init(const std::vector<tabular::ColumnSchema>& schema) {
    m_schema = schema;
    m_total_features = 0;

    // Pre-build cuDF AST expressions for scaling
    for (const auto& col : m_schema) {
        if (col.type == tabular::ColumnType::NUMERICAL || col.type == tabular::ColumnType::TIMESTAMP) {
            if (col.encoding == tabular::EncodingType::STANDARD_SCALE) {
                // (x - mean) / std
                std::string expr = "(x - " + std::to_string(col.mean) + ") / " + std::to_string(col.std + 1e-6f);
                m_scale_expressions.push_back(std::make_unique<cudf::ast::expression>(expr));
            } else if (col.encoding == tabular::EncodingType::MIN_MAX_SCALE) {
                // (x - min) / (max - min)
                std::string expr = "(x - " + std::to_string(col.min) + ") / (" + std::to_string(col.max - col.min + 1e-6f) + ")";
                m_scale_expressions.push_back(std::make_unique<cudf::ast::expression>(expr));
            } else {
                m_scale_expressions.push_back(nullptr); // No scaling needed
            }
        } else {
            m_scale_expressions.push_back(nullptr); // Not a numerical column
        }

        // Calculate output width
        if (col.type == tabular::ColumnType::IP_ADDRESS) m_total_features += 4;
        else if (col.type != tabular::ColumnType::IGNORE) m_total_features += 1;
    }
}

std::unique_ptr<cudf::column> CudfTabularPreprocessor::strings_to_cudf_column(const std::vector<std::string>& strings) {
    // Copy vector of strings to device memory
    std::vector<const char*> c_strings;
    std::vector<std::size_t> string_lengths;
    size_t total_length = 0;

    for (const auto& s : strings) {
        c_strings.push_back(s.c_str());
        string_lengths.push_back(s.length());
        total_length += s.length();
    }

    // cuDF requires a raw char* array on device
    auto stream_view = rmm::cuda_stream_view(m_cuda_stream);

    // Create strings column on device
    return cudf::strings::create_strings_column_from_scalar(
        cudf::string_scalar("", false), // Null scalar to start
        strings.size(), // Number of rows
        rmm::mr::get_current_device_resource(),
        stream_view
    ); // Actual fill from host vector needs specialized API call or temp buffer
}

// Custom GPU IP Parsing Logic (using cuDF strings functions)
std::unique_ptr<cudf::table> CudfTabularPreprocessor::parse_ip_cudf(const cudf::column& ip_column) {
    auto stream_view = rmm::cuda_stream_view(m_cuda_stream);

    // Example: Split "192.168.1.1" by "."
    // Returns a column of list_view<string>
    auto lists_of_octets = cudf::strings::split(ip_column, cudf::string_scalar("."));

    // Flatten to a single column of string octets
    auto octet_strings = cudf::strings::flatten_strings(lists_of_octets->view());

    // Convert string octets to integer column
    auto octets_int = cudf::strings::to_integers(octet_strings->view(), cudf::data_type{cudf::type_id::INT32}, stream_view);

    // Convert integer octets to float (0-255 -> 0.0-1.0)
    cudf::scalar<float> div_by_255(255.0f);
    auto octets_float = cudf::transform(octets_int->view(), cudf::ast::operation::DIV, div_by_255, stream_view);

    // Reshape back into 4 columns (or concatenate)
    // This is a simplified view. Actual implementation would involve grouping and transposing
    // to get 4 separate float columns for [IP1, IP2, IP3, IP4].
    // For this example, we'll just return a single column for simplicity

    // Returning a table of multiple columns (one for each octet) is complex with cuDF.
    // It's often easier to put all 4 octets into a list column or process as a single large column.

    return std::make_unique<cudf::table>(std::vector<std::unique_ptr<cudf::column>>{octets_float.release()});
}


void CudfTabularPreprocessor::process(const tabular::TableRow& raw_row, core::Tensor& dst) {
    // This path is inefficient. Copy to GPU then process small batch.
    // We expect process_batch to be used.
    // For now, delegate to batch with size 1.
    process_batch({raw_row}, dst);
}

void CudfTabularPreprocessor::process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) {
    if (rows.empty()) return;

    size_t batch_size = rows.size();
    auto stream_view = rmm::cuda_stream_view(m_cuda_stream);

    std::vector<std::unique_ptr<cudf::column>> processed_columns;

    for (size_t i = 0; i < m_schema.size(); ++i) {
        const auto& col_schema = m_schema[i];

        if (col_schema.type == tabular::ColumnType::IGNORE) continue;

        // 1. Extract Column Data for current batch (Host to Device)
        std::vector<std::string> col_strings_host;
        std::vector<float> col_floats_host;

        for(const auto& row : rows) {
            if (i < row.size()) {
                if (col_schema.type == tabular::ColumnType::NUMERICAL || col_schema.type == tabular::ColumnType::TIMESTAMP) {
                    try { col_floats_host.push_back(std::stof(row[i])); } catch(...) { col_floats_host.push_back(0.0f); }
                } else {
                    col_strings_host.push_back(row[i]);
                }
            } else {
                // Handle missing column data in row
                if (col_schema.type == tabular::ColumnType::NUMERICAL || col_schema.type == tabular::ColumnType::TIMESTAMP) col_floats_host.push_back(0.0f);
                else col_strings_host.push_back("");
            }
        }

        // 2. Process based on ColumnType
        if (col_schema.type == tabular::ColumnType::NUMERICAL || col_schema.type == tabular::ColumnType::TIMESTAMP) {
            // Create cuDF column from host floats
            std::unique_ptr<cudf::column> numeric_col = cudf::make_column_from_scalar(
                cudf::scalar<float>(0.0f),
                batch_size,
                stream_view
            ); // Replace with actual column creation from host data.
            // Placeholder: Assume data is copied to `numeric_col` from `col_floats_host`

            // Apply scaling expression
            if (m_scale_expressions[i]) {
                auto scaled_col = cudf::transform(numeric_col->view(), *m_scale_expressions[i], stream_view);
                processed_columns.push_back(std::move(scaled_col));
            } else {
                processed_columns.push_back(std::move(numeric_col));
            }

        } else if (col_schema.type == tabular::ColumnType::CATEGORICAL) {
            // Create string column for categories
            std::unique_ptr<cudf::column> string_col = strings_to_cudf_column(col_strings_host);

            // For LABEL_ENCODE, use cudf::strings::map_str_to_int or similar
            // For ONE_HOT, use cudf::one_hot_encode
            processed_columns.push_back(std::move(string_col)); // Placeholder
        } else if (col_schema.type == tabular::ColumnType::IP_ADDRESS) {
            // Create string column for IPs
            std::unique_ptr<cudf::column> ip_string_col = strings_to_cudf_column(col_strings_host);
            // Apply GPU IP parsing
            std::unique_ptr<cudf::table> ip_table = parse_ip_cudf(ip_string_col->view());

            // Need to flatten ip_table into individual columns and add to processed_columns
            // For now, adding a single placeholder column
            processed_columns.push_back(cudf::make_column_from_scalar(cudf::scalar<float>(0.0f), batch_size, stream_view));
            processed_columns.push_back(cudf::make_column_from_scalar(cudf::scalar<float>(0.0f), batch_size, stream_view));
            processed_columns.push_back(cudf::make_column_from_scalar(cudf::scalar<float>(0.0f), batch_size, stream_view));
            processed_columns.push_back(cudf::make_column_from_scalar(cudf::scalar<float>(0.0f), batch_size, stream_view));
        }
    }

    // 3. Concatenate all processed columns into a single flat tensor (Device -> Device)
    // The model expects a single contiguous float buffer [Batch * Features].
    auto output_table = cudf::concatenate(processed_columns, stream_view);

    // Extract the raw device pointer from the output table's column (assuming single output column)
    auto output_column = output_table->get_column(0);
    auto output_view = output_column.view();

    // Resize xInfer Tensor to hold the GPU data
    dst.resize({(int64_t)batch_size, (int64_t)m_total_features}, core::DataType::kFLOAT);

    // Get the raw device pointer from cuDF column
    // The data is already on GPU, we just need to bind it or copy it.
    // For zero-copy, you would ideally pass the cuDF column's device buffer handle directly to the inference engine.
    // For now, assume a copy to xInfer's GPU tensor.
    // This part is highly dependent on how xinfer::core::Tensor manages its device memory.

    // Placeholder for D2D copy:
    cudaMemcpyAsync(dst.data(), output_view.data<float>(), dst.size() * sizeof(float), cudaMemcpyDeviceToDevice, m_cuda_stream);
    cudaStreamSynchronize(m_cuda_stream); // Ensure completion
}

} // namespace xinfer::preproc

#endif // XINFER_HAS_CUDA