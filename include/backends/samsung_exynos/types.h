#pragma once

namespace xinfer::backends::exynos {

/**
 * @brief Exynos NPU Performance Preset
 * Controls the DVFS policies and core affinity.
 */
enum class EnnPowerMode {
    DEFAULT = 0,
    LOW_POWER = 1,       // Battery saving (background tasks)
    SUSTAINED = 2,       // Consistent frame times (Video/SIEM)
    BOOST = 3            // Maximum Frequency (Real-time/Aegis Sky)
};

/**
 * @brief Inference Priority
 * Determines preemption behavior on the NPU scheduler.
 */
enum class EnnPriority {
    NORMAL = 0,
    HIGH = 1,
    REALTIME = 2         // Hard deadline (Requires root/system privs)
};

/**
 * @brief Memory Layout preference
 * Exynos NPUs prefer specific alignments (usually 64-byte or 128-byte).
 */
enum class EnnMemoryMode {
    CACHED = 0,          // Standard alloc (Requires flush/invalidate)
    ION_CONTIGUOUS = 1   // ION/DMABUF (Zero-Copy)
};

} // namespace xinfer::backends::exynos