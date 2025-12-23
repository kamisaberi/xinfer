#include "scalers.h"
#include <cmath>
#include <algorithm>

namespace xinfer::preproc::tabular {

    // Small epsilon to prevent division by zero
    static constexpr float EPSILON = 1e-6f;

    // =================================================================================
    // Single Value Implementations
    // =================================================================================

    float scale_standard(float val, float mean, float std) {
        // Safety: If std is 0 (constant feature), return 0.0
        if (std::abs(std) < EPSILON) return 0.0f;
        return (val - mean) / std;
    }

    float scale_minmax(float val, float min, float max) {
        float denom = max - min;
        // Safety: If max == min, return 0.0
        if (std::abs(denom) < EPSILON) return 0.0f;
        return (val - min) / denom;
    }

    float scale_log1p(float val) {
        // std::log1p computes natural log of (1 + x)
        // Handles x=0 correctly (result 0)
        return std::log1p(std::max(0.0f, val));
    }

    // =================================================================================
    // Batch Implementations
    // =================================================================================

    void batch_scale_standard(const float* src, float* dst, size_t count, float mean, float std) {
        if (std::abs(std) < EPSILON) {
            // Feature has no variance, fill with 0
            std::fill(dst, dst + count, 0.0f);
            return;
        }

        float inv_std = 1.0f / std;

        // Compiler should auto-vectorize this loop (SSE/AVX/NEON)
        for (size_t i = 0; i < count; ++i) {
            dst[i] = (src[i] - mean) * inv_std;
        }
    }

    void batch_scale_minmax(const float* src, float* dst, size_t count, float min, float max) {
        float denom = max - min;

        if (std::abs(denom) < EPSILON) {
            std::fill(dst, dst + count, 0.0f);
            return;
        }

        float inv_denom = 1.0f / denom;

        for (size_t i = 0; i < count; ++i) {
            dst[i] = (src[i] - min) * inv_denom;
        }
    }

} // namespace xinfer::preproc::tabular