#pragma once

namespace xinfer::backends::openvino {

/**
 * @brief OpenVINO Device Target
 */
enum class DeviceType {
    CPU = 0,    // Standard Intel Core/Xeon
    GPU = 1,    // Intel Iris Xe / Arc Discrete Graphics
    NPU = 2,    // Intel Core Ultra (Meteor Lake) NPU
    AUTO = 3,   // Automatically select best device (CPU -> GPU -> NPU)
    HETERO = 4  // Split model across devices (e.g., CPU + GPU)
};

/**
 * @brief Performance Hint
 * OpenVINO optimizes the compilation/execution for specific goals.
 */
enum class PerformanceHint {
    LATENCY = 0,         // Minimize response time (Aegis Sky)
    THROUGHPUT = 1,      // Maximize FPS/Batching (Blackbox SIEM)
    CUMULATIVE_THROUGHPUT = 2 // For multi-device logic
};

/**
 * @brief Precision Mode for Inference
 */
enum class OvPrecision {
    FP32 = 0,
    FP16 = 1,  // Recommended for GPU/NPU
    INT8 = 2   // Requires quantized model (Post-Training Optimization)
};

} // namespace xinfer::backends::openvino