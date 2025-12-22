#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::rknn {

/**
 * @brief Configuration for Rockchip RKNN Backend
 */
struct RknnConfig {
    // Path to the compiled .rknn file
    std::string model_path;

    // NPU Core Selection
    RknnCoreMask core_mask = RknnCoreMask::AUTO;

    // Enable Zero-Copy path using DRM/DMA buffers?
    // Requires the input tensors to be allocated via the special
    // xinfer::core::MemoryType::CmaContiguous allocator.
    bool use_zero_copy = false;
    
    // If true, performs explicit sync/flush on memory buffers.
    // Necessary on some older kernels.
    bool mem_sync = true;

    // Number of threads for the internal scheduler (not CPU compute threads)
    // usually 1 is sufficient per model context.
    int num_threads = 1;
};

} // namespace xinfer::backends::rknn