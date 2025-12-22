#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::microchip {

/**
 * @brief Configuration for Microchip VectorBlox Backend
 */
struct VectorBloxConfig {
    // Path to the compiled model BLOB (.vnnx or .hex)
    // Produced by the VectorBlox SDK script (vnnx-to-blob)
    std::string model_path;

    // The core architecture currently programmed on the FPGA fabric
    VbxCoreType core_type = VbxCoreType::V1000;

    // Clock frequency of the VectorBlox IP in Hz (e.g., 100000000 for 100MHz)
    // Required for accurate timeout calculations.
    uint32_t clock_frequency = 100000000;

    // Use specific DDR memory region for tensor buffers?
    // If 0, uses standard system malloc/CMA.
    uintptr_t memory_base_addr = 0;
    
    // Size of the dedicated memory region
    size_t memory_size = 0;
};

} // namespace xinfer::backends::microchip