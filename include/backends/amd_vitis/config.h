#pragma once

#include <string>
#include <vector>
#include "types.h"

namespace xinfer::backends::vitis {

    /**
     * @brief Configuration for AMD/Xilinx Vitis AI Backend
     */
    struct VitisConfig {
        // Path to the compiled model artifact (.xmodel)
        std::string model_path;

        // Optional: Path to a specific bitstream (.xclbin) to program onto the FPGA.
        // If empty, assumes the FPGA is already programmed via boot or xmutil.
        std::string xclbin_path;

        // Expected DPU Architecture (Safety check)
        DpuArch target_arch = DpuArch::DPUCZDX8G;

        // Number of DPU runners (threads) to spawn.
        // Increasing this improves throughput for batch processing on SIEM.
        int num_runners = 1;

        // Memory strategy for input/output tensors.
        // Use ZERO_COPY_DMABUF for Aegis Sky low-latency requirements.
        MemoryStrategy memory_strategy = MemoryStrategy::COPY_ALWAYS;

        // Enable profiling dump (vitis_ai_profiler)
        bool enable_profiling = false;
    };

} // namespace xinfer::backends::vitis