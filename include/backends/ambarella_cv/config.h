#pragma once

#include <string>
#include <cstdint>

namespace xinfer::backends::ambarella {

    /**
     * @brief Configuration for the CVFlow Engine
     */
    struct AmbarellaConfig {
        // Path to the compiled binary (.cavalry file)
        std::string model_path;

        // Which Vector Processor to use (CV chips often have multiple)
        // 0 = VP0, 1 = VP1
        int vp_instance_id = 0;

        // Ambarella uses a specialized memory pool (CMA).
        // Define how much memory to reserve for input/output buffers.
        size_t memory_pool_size = 1024 * 1024 * 64; // 64MB default

        // Priority level for the OS scheduler (Real-time tracking for Aegis Sky)
        int priority_level = 99;

        // Enable debug logs from the Cavalry firmware
        bool verbose_firmware_logs = false;
    };

} // namespace xinfer::backends::ambarella