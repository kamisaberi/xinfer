#pragma once

namespace xinfer::backends::ryzen_ai {

    /**
     * @brief NPU Power/Performance Profile
     * Ryzen AI NPUs can switch context to prioritize throughput or latency.
     */
    enum class XdnaProfile {
        DEFAULT = 0,
        LATENCY_OPTIMIZED = 1, // Single-batch, fast response
        THROUGHPUT_OPTIMIZED = 2, // High batch size, higher latency
        LOW_POWER = 3 // Throttle NPU clock for battery saving
    };

    /**
     * @brief Execution Provider Type
     * Ryzen AI can be accessed via the Vitis AI Execution Provider (ONNX)
     * or the native XRT (Xilinx Runtime) API.
     */
    enum class RuntimeType {
        VITIS_AI_EP = 0, // Uses ONNX Runtime with Vitis AI EP (Easier, Standard)
        NATIVE_XRT = 1   // Uses raw XRT/VART (Maximum control, requires .xmodel)
    };

} // namespace xinfer::backends::ryzen_ai