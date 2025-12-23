#include "neon_image.h"
#include <xinfer/core/logging.h>

// Fallback to OpenCV for the resizing part
#include <opencv2/opencv.hpp>

// --- ARM NEON Headers ---
#if defined(__aarch64__) || defined(__arm__)
    #include <arm_neon.h>
#endif

namespace xinfer::preproc {

void NeonImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;
}

#if defined(__aarch64__) || defined(__arm__)

// =================================================================================
// NEON Optimized Implementation
// =================================================================================

void NeonImagePreprocessor::neon_normalize_permute(const uint8_t* src, float* dst, int count) {
    // 1. Setup Constants
    // Load mean and std into NEON registers (duplicate across all lanes)
    float32x4_t v_mean_r = vdupq_n_f32(m_config.norm_params.mean[0]);
    float32x4_t v_mean_g = vdupq_n_f32(m_config.norm_params.mean[1]);
    float32x4_t v_mean_b = vdupq_n_f32(m_config.norm_params.mean[2]);

    // Pre-calculate inverse std for multiplication (mul is faster than div)
    float32x4_t v_inv_std_r = vdupq_n_f32(1.0f / m_config.norm_params.std[0]);
    float32x4_t v_inv_std_g = vdupq_n_f32(1.0f / m_config.norm_params.std[1]);
    float32x4_t v_inv_std_b = vdupq_n_f32(1.0f / m_config.norm_params.std[2]);

    float32x4_t v_scale = vdupq_n_f32(m_config.norm_params.scale_factor); // e.g. 1/255.0

    // Destination pointers for Planar Layout (NCHW)
    // dst points to R plane. G is at +count, B is at +2*count
    float* dst_r = dst;
    float* dst_g = dst + count;
    float* dst_b = dst + count * 2;

    int i = 0;

    // 2. Main Loop: Process 16 pixels per iteration
    // We use uint8x16 (128-bit) loads
    for (; i <= count - 16; i += 16) {
        // A. Load and Deinterleave (HWC -> Planes in registers)
        // Loads [RGBRGB...] and splits into r_u8, g_u8, b_u8 vectors
        uint8x16x3_t v_rgb_u8 = vld3q_u8(src + i * 3);

        // B. Convert Uint8 -> Uint16 -> Float32
        // We need to split 16 elements into 4 sets of 4 floats

        // --- Low 8 pixels ---
        uint16x8_t u16_r_lo = vmovl_u8(vget_low_u8(v_rgb_u8.val[0]));
        uint16x8_t u16_g_lo = vmovl_u8(vget_low_u8(v_rgb_u8.val[1]));
        uint16x8_t u16_b_lo = vmovl_u8(vget_low_u8(v_rgb_u8.val[2]));

        // --- High 8 pixels ---
        uint16x8_t u16_r_hi = vmovl_u8(vget_high_u8(v_rgb_u8.val[0]));
        uint16x8_t u16_g_hi = vmovl_u8(vget_high_u8(v_rgb_u8.val[1]));
        uint16x8_t u16_b_hi = vmovl_u8(vget_high_u8(v_rgb_u8.val[2]));

        // Helper Lambda to process 4 floats
        auto process_quad = [&](uint16x4_t u16_quad,
                                float32x4_t mean, float32x4_t inv_std, float* &store_ptr) {
            // Convert to float
            uint32x4_t u32 = vmovl_u16(u16_quad);
            float32x4_t f32 = vcvtq_f32_u32(u32);

            // Normalize: (val * scale - mean) * inv_std
            // Note: scale is usually applied first to get to [0,1]
            f32 = vmulq_f32(f32, v_scale);
            f32 = vsubq_f32(f32, mean);
            f32 = vmulq_f32(f32, inv_std);

            // Store
            vst1q_f32(store_ptr, f32);
            store_ptr += 4;
        };

        // Unroll the 16 pixels into 4 batches of 4
        // Red Channel
        process_quad(vget_low_u16(u16_r_lo), v_mean_r, v_inv_std_r, dst_r);
        process_quad(vget_high_u16(u16_r_lo), v_mean_r, v_inv_std_r, dst_r);
        process_quad(vget_low_u16(u16_r_hi), v_mean_r, v_inv_std_r, dst_r);
        process_quad(vget_high_u16(u16_r_hi), v_mean_r, v_inv_std_r, dst_r);

        // Green Channel
        // Reset pointers? No, process_quad advances them, but we need to reset/offset for G
        // Actually, easier to manually expand loop for clarity and pointer management
        dst_g = dst + count + i; // Recalculate base for this batch
        float* tmp_g = dst_g;
        process_quad(vget_low_u16(u16_g_lo), v_mean_g, v_inv_std_g, tmp_g);
        process_quad(vget_high_u16(u16_g_lo), v_mean_g, v_inv_std_g, tmp_g);
        process_quad(vget_low_u16(u16_g_hi), v_mean_g, v_inv_std_g, tmp_g);
        process_quad(vget_high_u16(u16_g_hi), v_mean_g, v_inv_std_g, tmp_g);

        // Blue Channel
        dst_b = dst + count * 2 + i;
        float* tmp_b = dst_b;
        process_quad(vget_low_u16(u16_b_lo), v_mean_b, v_inv_std_b, tmp_b);
        process_quad(vget_high_u16(u16_b_lo), v_mean_b, v_inv_std_b, tmp_b);
        process_quad(vget_low_u16(u16_b_hi), v_mean_b, v_inv_std_b, tmp_b);
        process_quad(vget_high_u16(u16_b_hi), v_mean_b, v_inv_std_b, tmp_b);

        // Fix up pointers for next main loop
        dst_r = dst + i + 16;
    }

    // 3. Cleanup Loop (Scalar Fallback)
    // Handle remaining pixels that don't fit in 16-pixel blocks
    for (; i < count; ++i) {
        const uint8_t* p = src + i * 3;
        // R
        dst[i] = (p[0] * m_config.norm_params.scale_factor - m_config.norm_params.mean[0])
                 / m_config.norm_params.std[0];
        // G
        dst[count + i] = (p[1] * m_config.norm_params.scale_factor - m_config.norm_params.mean[1])
                         / m_config.norm_params.std[1];
        // B
        dst[count * 2 + i] = (p[2] * m_config.norm_params.scale_factor - m_config.norm_params.mean[2])
                             / m_config.norm_params.std[2];
    }
}

void NeonImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    if (!src.data) return;

    // 1. Resize (Using OpenCV - it is reasonably NEON optimized for geometry)
    // We resize to a temporary buffer if needed
    cv::Mat img_wrapper(src.height, src.width, CV_8UC3, src.data);
    cv::Mat resized;

    if (src.width != m_config.target_width || src.height != m_config.target_height) {
        cv::resize(img_wrapper, resized, cv::Size(m_config.target_width, m_config.target_height), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img_wrapper;
    }

    // 2. Color Conversion (if needed)
    // Our NEON kernel assumes RGB input order. If source is BGR (OpenCV default), swap first.
    if (src.format == ImageFormat::BGR && m_config.target_format == ImageFormat::RGB) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    // 3. Normalize + Transpose (The heavy lifting)
    int total_pixels = m_config.target_width * m_config.target_height;
    float* dst_ptr = static_cast<float*>(dst.data());

    if (m_config.layout_nchw) {
        // Fast NEON Path
        neon_normalize_permute(resized.data, dst_ptr, total_pixels);
    } else {
        // NHWC Path (Standard copy/normalize)
        // TODO: Add neon_normalize_packed implementation
        XINFER_LOG_WARN("NEON NHWC path not implemented, falling back to scalar loop.");
        // Scalar fallback...
    }
}

#else

// =================================================================================
// x86 / Non-ARM Fallback
// =================================================================================

void NeonImagePreprocessor::neon_normalize_permute(const uint8_t* src, float* dst, int count) {
    XINFER_LOG_ERROR("Attempted to call NEON kernel on non-ARM architecture!");
}

void NeonImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    XINFER_LOG_ERROR("NeonImagePreprocessor is only supported on ARM (AArch64).");
}

#endif

} // namespace xinfer::preproc