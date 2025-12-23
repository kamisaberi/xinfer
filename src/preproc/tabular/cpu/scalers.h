#pragma once

#include <cstddef>

namespace xinfer::preproc::tabular {

    /**
     * @brief Numerical Scaling Utilities
     *
     * Functions to normalize raw float values.
     * Optimized for SIEM log features (Packet Size, Duration, etc.).
     */

    // --- Single Value (for Row-by-Row processing) ---

    /**
     * @brief Standard Scaler (Z-Score)
     * Formula: (x - mean) / std
     */
    float scale_standard(float val, float mean, float std);

    /**
     * @brief Min-Max Scaler
     * Formula: (x - min) / (max - min)
     * Result is usually in [0, 1] range.
     */
    float scale_minmax(float val, float min, float max);

    /**
     * @brief Log Scaler
     * Formula: log1p(x) -> ln(x + 1)
     * Useful for highly skewed data like Byte Counts.
     */
    float scale_log1p(float val);


    // --- Batch Processing (SIMD Friendly) ---

    /**
     * @brief Apply Standard Scaling to an array.
     */
    void batch_scale_standard(const float* src, float* dst, size_t count, float mean, float std);

    /**
     * @brief Apply Min-Max Scaling to an array.
     */
    void batch_scale_minmax(const float* src, float* dst, size_t count, float min, float max);

} // namespace xinfer::preproc::tabular