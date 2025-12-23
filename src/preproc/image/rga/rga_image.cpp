#ifdef XINFER_HAS_RGA

#include "rga_image.h"
#include <rga/RgaApi.h> // Standard Rockchip header
#include <xinfer/core/logging.h>

namespace xinfer::preproc {

    void RgaImagePreprocessor::init(const ImagePreprocConfig& config) {
        m_config = config;
        // RGA initialization usually happens globally or lazily
    }

    void RgaImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
        // RGA requires physical addresses or dma_buf fds for maximum speed.
        // Here we assume virtual address mode (slower but works with malloc).

        rga_buffer_t rga_src = {};
        rga_src.vir_addr = src.data;
        rga_src.width = src.width;
        rga_src.height = src.height;
        rga_src.wstride = src.width; // Padding alignment
        rga_src.hstride = src.height;

        // Map Format
        if (src.format == ImageFormat::RGB) rga_src.format = RK_FORMAT_RGB_888;
        else if (src.format == ImageFormat::NV12) rga_src.format = RK_FORMAT_YCbCr_420_SP; // Critical for Video

        rga_buffer_t rga_dst = {};
        rga_dst.vir_addr = dst.data(); // Writing directly to Tensor memory
        rga_dst.width = m_config.target_width;
        rga_dst.height = m_config.target_height;
        rga_dst.wstride = m_config.target_width;
        rga_dst.hstride = m_config.target_height;
        rga_dst.format = RK_FORMAT_RGB_888; // RGA outputs packed RGB

        // Perform Resize
        int ret = c_RkRgaBlit(&rga_src, &rga_dst, NULL);

        if (ret < 0) {
            XINFER_LOG_ERROR("RGA Blit failed: " + std::to_string(ret));
            // Fallback to CPU if needed
        }

        // NOTE: RGA outputs NHWC (Packed).
        // If Model needs NCHW, we still need a CPU permute step or a custom NPU permutation layer.
        // Ideally, compile your RKNN model with 'force_builtin_perm=True' so the NPU accepts NHWC.
    }

}

#endif // XINFER_HAS_RGA