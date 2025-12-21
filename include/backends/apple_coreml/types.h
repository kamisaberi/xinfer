#pragma once

namespace xinfer::backends::coreml {

    /**
     * @brief Core ML Compute Units
     * Determines which hardware runs the model.
     */
    enum class ComputeUnit {
        ALL = 0,               // Use Neural Engine, GPU, and CPU (Best for performance)
        CPU_AND_GPU = 1,       // Skip Neural Engine (Useful if ANE is unsupported for ops)
        CPU_ONLY = 2,          // Force CPU execution (Best for debugging)
        CPU_AND_NEURAL_ENGINE = 3 // Skip GPU (Keeps GPU free for rendering)
    };

    /**
     * @brief Precision strategy for Core ML
     */
    enum class PrecisionType {
        FLOAT32 = 0,
        FLOAT16 = 1 // Allow Core ML to reduce precision for ANE speedup
    };

} // namespace xinfer::backends::coreml