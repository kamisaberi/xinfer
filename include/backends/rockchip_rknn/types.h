#pragma once

#include <cstdint>

namespace xinfer::backends::rknn {

/**
 * @brief RKNN NPU Core Mask
 * Controls which NPU cores are used for inference.
 * RK3588 has 3 cores (0, 1, 2).
 */
enum class RknnCoreMask : uint32_t {
    AUTO = 0,         // Let the driver decide
    CORE_0 = 1,       // Force Core 0
    CORE_1 = 2,       // Force Core 1
    CORE_2 = 4,       // Force Core 2 (RK3588 only)
    CORE_0_1 = 3,     // Cores 0 and 1 combined
    CORE_0_1_2 = 7    // All cores combined (Highest throughput/latency trade-off)
};

/**
 * @brief Tensor Data Format
 * RKNN models usually expect NHWC for images.
 */
enum class RknnDataFormat {
    NCHW = 0,
    NHWC = 1,
    UNDEFINED = 2
};

} // namespace xinfer::backends::rknn