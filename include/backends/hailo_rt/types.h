#pragma once

namespace xinfer::backends::hailo {

/**
 * @brief Hailo Device Connection Interface
 */
enum class DeviceInterface {
    ANY = 0,      // Pick the first available device (PCIe or USB)
    PCIE = 1,     // Force PCIe (M.2 Key M/E)
    USB = 2,      // Force USB (Hailo-8L Stick)
    ETHERNET = 3  // Hailo-15 / Networked mode
};

/**
 * @brief Input/Output Stream Format
 * Hailo hardware usually expects quantized UINT8/INT8, but the runtime
 * can perform automatic transform from Float32 on the host CPU.
 */
enum class StreamFormat {
    AUTO = 0,           // Let HailoRT decide based on the HEF file
    USER_FLOAT32 = 1,   // Host sends Floats, runtime quantizes
    USER_UINT8 = 2      // Host sends raw UINT8 (Fastest, zero-copy possible)
};

} // namespace xinfer::backends::hailo