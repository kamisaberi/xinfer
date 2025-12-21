#pragma once

namespace xinfer::backends::google_tpu {

    /**
     * @brief Edge TPU Device Type
     * Specifies which interface the TPU is connected through.
     */
    enum class TpuDeviceType {
        ANY = 0, // Pick the first available TPU
        USB = 1, // Only look for USB Accelerators
        PCI = 2  // Only look for PCIe/M.2 cards
    };

    /**
     * @brief TPU Power Frequency
     * Note: High performance mode requires adequate cooling.
     */
    enum class TpuPerformanceMode {
        STANDARD = 0,
        HIGH = 1, // "Max" frequency (requires active cooling)
        LOW = 2
    };

} // namespace xinfer::backends::google_tpu