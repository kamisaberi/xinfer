#pragma once

namespace xinfer::backends::vitis {

    /**
     * @brief DPU (Deep Learning Processing Unit) Architecture targets.
     * Used to validate if the loaded .xmodel matches the FPGA bitstream.
     */
    enum class DpuArch {
        UNKNOWN = 0,
        DPUCZDX8G = 1, // Edge: Zynq MPSoC, Kria KV260/KR260 (Most common)
        DPUCADF8H = 2, // Cloud: Alveo U200/U250 (High Throughput)
        DPUCAHX8H = 3, // Cloud: Alveo U50/U280 (HBM High Bandwidth)
        DPUCVDX8G = 4  // Edge: Versal AI Core (Aegis Sky high-end target)
    };

    /**
     * @brief Zero-Copy Tensor strategy.
     * Defines how xInfer handles data movement between CPU and FPGA.
     */
    enum class MemoryStrategy {
        COPY_ALWAYS = 0,    // Standard memcpy (Safe, works with any memory)
        ZERO_COPY_DMABUF = 1, // Use dma_buf/udmabuf (Requires CMA allocator)
        SHARED_VIRTUAL = 2  // For platforms with cache-coherent shared memory (Versal/Apple)
    };

} // namespace xinfer::backends::vitis