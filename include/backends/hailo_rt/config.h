#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::hailo {

/**
 * @brief Configuration for HailoRT Backend
 */
struct HailoConfig {
    // Path to the compiled .hef file
    std::string model_path;

    // Device selection strategy
    DeviceInterface device_type = DeviceInterface::ANY;
    
    // Specific device ID (e.g., "0000:03:00.0" for PCIe)
    // If empty, uses the first device found.
    std::string device_id;

    // Timeout for inference in milliseconds
    uint32_t timeout_ms = 1000;

    // Batch size. MUST match the batch size the HEF was compiled with.
    // Hailo does not support dynamic batching at runtime easily.
    uint16_t batch_size = 1;

    // Preferred format for data exchange with Host CPU.
    // Set to USER_UINT8 if you are passing raw image bytes for max speed.
    StreamFormat input_format = StreamFormat::USER_FLOAT32;
    StreamFormat output_format = StreamFormat::USER_FLOAT32;
    
    // Enable scheduler (allows multiple models on one chip)
    bool use_multiplexer = false;
};

} // namespace xinfer::backends::hailo