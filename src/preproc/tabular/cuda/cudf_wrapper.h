#pragma once

#include <xinfer/preproc/tabular/tabular_preprocessor.h>
#include <xinfer/preproc/tabular/types.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <memory>

// Include cuDF headers only if CUDA is enabled
#ifdef XINFER_HAS_CUDA
// Forward declare cuDF types to avoid heavy includes in a header
namespace cudf {
    class table_view;
    class table;
    class column;
    namespace ast {
        class expression;
    }
}
#endif

namespace xinfer::preproc {

/**
 * @brief CUDA cuDF Tabular Preprocessor
 *
 * Offloads tabular data preprocessing (parsing, scaling, encoding) to the GPU
 * using NVIDIA's RAPIDS cuDF library.
 *
 * Designed for high-throughput batch processing of structured data (SIEM logs).
 *
 * Requirements:
 * - NVIDIA GPU with CUDA
 * - cuDF library installed (part of RAPIDS)
 */
class CudfTabularPreprocessor : public ITabularPreprocessor {
public:
    CudfTabularPreprocessor();
    ~CudfTabularPreprocessor() override;

    void init(const std::vector<tabular::ColumnSchema>& schema) override;

    /**
     * @brief Process a single row (Host -> Device).
     *
     * @note This path is not optimized for single rows (H2D overhead).
     * Use `process_batch` for best performance.
     */
    void process(const tabular::TableRow& raw_row, core::Tensor& dst) override;

    /**
     * @brief Batch Process rows on GPU.
     *
     * @param rows Vector of raw string rows (Host).
     * @param dst Output Tensor (Device memory, Float32).
     */
    void process_batch(const std::vector<tabular::TableRow>& rows, core::Tensor& dst) override;

    size_t get_output_width() const override { return m_total_features; }

private:
#ifdef XINFER_HAS_CUDA
    std::vector<tabular::ColumnSchema> m_schema;
    size_t m_total_features = 0;

    // cuDF expressions for scaling/encoding
    // These are compiled once and applied to columns
    std::vector<std::unique_ptr<cudf::ast::expression>> m_scale_expressions;

    // Cached CUDA streams for asynchronous operations
    cudaStream_t m_cuda_stream = nullptr;

    // Helper to convert host strings to cuDF column of strings
    std::unique_ptr<cudf::column> strings_to_cudf_column(const std::vector<std::string>& strings);

    // Helper for IP address parsing on GPU
    std::unique_ptr<cudf::table> parse_ip_cudf(const cudf::column& ip_column);
#endif
};

} // namespace xinfer::preproc