#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>

// Xilinx Runtime Headers
// (Wrapped in ifdef to allow compilation on non-FPGA machines)
#ifdef XINFER_ENABLE_VITIS
#include <experimental/xrt_kernel.h>
#include <experimental/xrt_bo.h>
#endif

namespace xinfer::preproc {

    /**
     * @brief FPGA Image Preprocessor (XRT/Vitis)
     *
     * Offloads preprocessing to a PL (Programmable Logic) kernel.
     * This effectively implements a hardware accelerator for:
     * Resize -> Color Conv -> Normalize.
     *
     * Requires:
     * 1. An XRT-compatible bitstream loaded (.xclbin).
     * 2. A specific kernel in the bitstream (default: "pp_pipeline_accel").
     */
    class FpgaImagePreprocessor : public IImagePreprocessor {
    public:
        FpgaImagePreprocessor();
        ~FpgaImagePreprocessor() override;

        void init(const ImagePreprocConfig& config) override;

        /**
         * @brief Process image using FPGA Hardware.
         *
         * 1. Copies Host Image -> FPGA DDR (Input BO).
         * 2. Triggers PL Kernel.
         * 3. Result is written to Output BO (which should be mapped to the Inference Tensor).
         */
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
#ifdef XINFER_ENABLE_VITIS
        xrt::device m_device;
        xrt::kernel m_kernel;
        xrt::run m_run;

        // Persistent Buffers to avoid reallocation
        xrt::bo m_bo_input;
        xrt::bo m_bo_output;
        size_t m_input_capacity = 0;
        size_t m_output_capacity = 0;
#endif

        ImagePreprocConfig m_config;
        std::string m_kernel_name = "pp_pipeline_accel"; // Standard name in Vitis AI examples
    };

} // namespace xinfer::preproc