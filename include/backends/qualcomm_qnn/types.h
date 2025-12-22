#pragma once

namespace xinfer::backends::qnn {

/**
 * @brief QNN Backend Type
 * Which hardware block to target.
 */
enum class QnnBackendType {
    CPU = 0, // Reference implementation (Slow)
    GPU = 1, // Adreno GPU (OpenCL based)
    DSP = 2, // Hexagon DSP (Legacy)
    HTP = 3  // Hexagon Tensor Processor (Modern NPU - Best Performance)
};

/**
 * @brief HTP Performance Mode
 * Controls the clock frequency and voltage voting for the DSP/NPU.
 */
enum class HtpPerformanceMode {
    BURST = 0,          // Max frequency (Short duration, high heat)
    SUSTAINED_HIGH = 1, // High consistency (Aegis Sky target)
    BALANCED = 2,       // Standard usage
    LOW_POWER = 3,      // Battery saving (Blackbox SIEM edge node)
    POWER_SAVER = 4
};

/**
 * @brief Precision / Quantization Type
 */
enum class QnnPrecision {
    FLOAT32 = 0,
    FLOAT16 = 1,
    QUANTIZED_U8 = 2,
    QUANTIZED_U16 = 3
};

} // namespace xinfer::backends::qnn