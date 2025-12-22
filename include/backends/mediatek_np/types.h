#pragma once

namespace xinfer::backends::mediatek {

/**
 * @brief NeuroPilot Execution Preference
 * Controls how the scheduler assigns work to the APU cores.
 */
enum class NeuronPreference {
    PREFER_LOW_POWER = 0,   // Minimize battery drain
    PREFER_FAST_SINGLE_ANSWER = 1, // Lowest Latency (Aegis Sky)
    PREFER_SUSTAINED_SPEED = 2,    // Best consistency (Thermal throttling mitigation)
    PREFER_ULTRA_PERFORMANCE = 3   // Max clocks (Requires active cooling)
};

/**
 * @brief APU Device Generation
 * Used to validate model compatibility.
 */
enum class ApuGeneration {
    UNKNOWN = 0,
    APU_v1 = 1, // Helio P60/P90 era
    APU_v2 = 2, // Dimensity 800/1000
    APU_v3 = 3, // Dimensity 1200
    APU_v4 = 4  // Dimensity 9000 / Genio 1200
};

} // namespace xinfer::backends::mediatek