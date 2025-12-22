#pragma once

namespace xinfer::backends::microchip {

/**
 * @brief VectorBlox Core Architecture
 * Defines the parallelism of the loaded IP Core.
 * Used to calculate expected throughput and buffer requirements.
 */
enum class VbxCoreType {
    UNKNOWN = 0,
    V250 = 1,    // Smallest, low power (IoT)
    V500 = 2,
    V1000 = 3,   // Standard Edge
    V2000 = 4,   // High Performance (Aegis Sky target)
    V4000 = 5    // Maximum Throughput
};

/**
 * @brief Execution Mode
 */
enum class VbxExecutionMode {
    SYNCHRONOUS = 0, // Block CPU until inference completes
    ASYNC_IRQ = 1    // Use Interrupts (Requires Kernel Driver support)
};

} // namespace xinfer::backends::microchip