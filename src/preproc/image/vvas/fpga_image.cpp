#include "fpga_image.h"
#include <xinfer/core/logging.h>
#include <cstring>

namespace xinfer::preproc {

FpgaImagePreprocessor::FpgaImagePreprocessor() {
#ifdef XINFER_ENABLE_VITIS
    // 1. Open Device (Index 0 is usually the embedded platform itself)
    try {
        m_device = xrt::device(0);

        // Load the PL Kernel (Preprocessing Accelerator)
        // Assumes the xclbin is already loaded by the OS or the Inference Backend.
        // We attach to the existing UUID.
        auto uuid = m_device.get_xclbin_uuid();
        m_kernel = xrt::kernel(m_device, uuid, m_kernel_name);

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Failed to initialize FPGA Preprocessor XRT: " + std::string(e.what()));
    }
#endif
}

FpgaImagePreprocessor::~FpgaImagePreprocessor() {
    // XRT RAII handles cleanup
}

void FpgaImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;
}

void FpgaImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
#ifdef XINFER_ENABLE_VITIS
    if (!src.data) return;

    size_t src_size = src.width * src.height * 3; // Assuming RGB
    size_t dst_size = m_config.target_width * m_config.target_height * 3 * sizeof(float); // NCHW Float

    // 1. Allocate/Resize Input Buffer (Host -> FPGA)
    if (m_input_capacity < src_size) {
        // Allocate buffer in Bank 0
        m_bo_input = xrt::bo(m_device, src_size, m_kernel.group_id(0));
        m_input_capacity = src_size;
    }

    // 2. Allocate/Resize Output Buffer (FPGA -> Inference)
    if (m_output_capacity < dst_size) {
        // Allocate buffer in Bank 0 (or match DPU bank requirements)
        m_bo_output = xrt::bo(m_device, dst_size, m_kernel.group_id(1));
        m_output_capacity = dst_size;
    }

    // 3. Write Data (H2D)
    // TODO: For zero-copy, map src.data DMABUF fd instead of writing
    m_bo_input.write(src.data);
    m_bo_input.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // 4. Set Kernel Arguments
    // Registers depend on HLS kernel signature.
    // Standard signature: (in_ptr, out_ptr, height, width, stride, mean/scale...)
    try {
        auto run = m_kernel(m_bo_input, m_bo_output,
                            src.height, src.width,
                            m_config.target_height, m_config.target_width,
                            m_config.norm_params.scale_factor); // Simplified args

        // 5. Wait for Completion
        run.wait();

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("FPGA Preproc Kernel Execution Failed: " + std::string(e.what()));
        return;
    }

    // 6. Map Output to Tensor
    // IMPORTANT: To avoid Device->Host->Device copy (FPGA -> CPU -> DPU),
    // we should really expose the physical address of m_bo_output to the Inference Engine.

    if (dst.memory_type() == core::MemoryType::CmaContiguous) {
        // Optimal Path: Just pass physical address handle (if tensor supports it)
        // dst.set_physical_address(m_bo_output.address());
    } else {
        // Fallback: Copy back to CPU (Slow)
        m_bo_output.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        m_bo_output.read(dst.data());
    }

#else
    XINFER_LOG_ERROR("Vitis support not enabled in build.");
#endif
}

} // namespace xinfer::preproc