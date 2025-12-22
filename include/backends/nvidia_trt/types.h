#pragma once

#include <cstdint>

namespace xinfer::backends::nvidia {

/**
 * @brief TensorRT Precision Mode
 * Determines the math precision used by the engine.
 */
enum class TrtPrecision {
    FP32 = 0,
    FP16 = 1, // Requires Tensor Cores (Volta+)
    INT8 = 2, // Requires Calibration
    FP8  = 3  // Requires Ada Lovelace / Hopper
};

/**
 * @brief DLACore (Deep Learning Accelerator)
 * Specific to Jetson Xavier/Orin devices.
 */
enum class DlaCore {
    GPU_FALLBACK = -1, // Use standard GPU cores
    DEFAULT = 0,       // Use DLA Core 0
    CORE_0 = 0,
    CORE_1 = 1         // Only available on larger Jetsons
};

} // namespace xinfer::backends::nvidia