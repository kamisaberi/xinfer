#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::google_tpu {

    /**
     * @brief Configuration for Google Edge TPU Backend
     */
    struct EdgeTpuConfig {
        // Path to the compiled .tflite file
        // MUST be compiled with 'edgetpu_compiler'
        std::string model_path;

        // Which specific TPU device to use
        TpuDeviceType device_type = TpuDeviceType::ANY;

        // Index of the device if multiple are present (0, 1, 2...)
        int device_index = 0;

        // Number of CPU threads for ops that fall back to CPU
        // (Ops not supported by TPU or not compiled)
        int cpu_threads = 1;

        // Set TPU clock speed (only affects USB accelerators mostly)
        TpuPerformanceMode performance = TpuPerformanceMode::STANDARD;

        // Optional: Path to the 'libedgetpu.so.1' shared library
        // If empty, uses the system default path.
        std::string libedgetpu_path;
    };

} // namespace xinfer::backends::google_tpu