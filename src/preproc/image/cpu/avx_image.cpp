#include "avx_image.h"
#include <xinfer/core/logging.h>

// Fallback to OpenCV for the resizing part
#include <opencv2/opencv.hpp>

// --- AVX Headers ---
#if defined(__AVX2__)
    #include <immintrin.h>
#endif

namespace xinfer::preproc {

void AvxImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;
}

#if defined(__AVX2__)

// =================================================================================
// AVX2 Optimized Implementation
// =================================================================================

void AvxImagePreprocessor::avx_normalize_permute(const uint8_t* src, float* dst, int count) {
    // 1. Setup Constants
    // -------------------------------------------------------------------------
    // Mean & Std (Broadcast to all 8 lanes)
    __m256 v_mean_r = _mm256_set1_ps(m_config.norm_params.mean[0]);
    __m256 v_mean_g = _mm256_set1_ps(m_config.norm_params.mean[1]);
    __m256 v_mean_b = _mm256_set1_ps(m_config.norm_params.mean[2]);

    // Inverse Std (Optimization: multiply is faster than divide)
    __m256 v_inv_std_r = _mm256_set1_ps(1.0f / m_config.norm_params.std[0]);
    __m256 v_inv_std_g = _mm256_set1_ps(1.0f / m_config.norm_params.std[1]);
    __m256 v_inv_std_b = _mm256_set1_ps(1.0f / m_config.norm_params.std[2]);

    __m256 v_scale = _mm256_set1_ps(m_config.norm_params.scale_factor); // 1/255.0

    // Gather Indices:
    // We want to load 8 pixels. Each pixel is 3 bytes (RGB).
    // Pixel 0 starts at 0, Pixel 1 at 3, Pixel 2 at 6...
    // AVX Gather loads 32-bit integers. We load 4 bytes starting at these offsets,
    // then mask out the upper 24 bytes to isolate the uint8 value.
    __m256i v_idx_r = _mm256_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21);
    __m256i v_idx_g = _mm256_setr_epi32(1, 4, 7, 10, 13, 16, 19, 22);
    __m256i v_idx_b = _mm256_setr_epi32(2, 5, 8, 11, 14, 17, 20, 23);

    // Mask to isolate the lowest byte (uint8) after 32-bit gather
    __m256i v_mask_u8 = _mm256_set1_epi32(0xFF);

    // Destination pointers for Planar Layout (NCHW)
    float* dst_r = dst;
    float* dst_g = dst + count;
    float* dst_b = dst + count * 2;

    int i = 0;

    // 2. Main Loop: Process 8 pixels per iteration
    // -------------------------------------------------------------------------
    for (; i <= count - 8; i += 8) {
        const int* src_int_ptr = reinterpret_cast<const int*>(src + i * 3);

        // A. Gather (De-interleave)
        // _mm256_i32gather_epi32(base_addr, index, scale)
        // Loads [R G B R], [G B R G]... as 32-bit integers
        __m256i raw_r = _mm256_i32gather_epi32(src_int_ptr, v_idx_r, 1);
        __m256i raw_g = _mm256_i32gather_epi32(src_int_ptr, v_idx_g, 1);
        __m256i raw_b = _mm256_i32gather_epi32(src_int_ptr, v_idx_b, 1);

        // B. Mask to uint8 (0-255) stored in int32
        raw_r = _mm256_and_si256(raw_r, v_mask_u8);
        raw_g = _mm256_and_si256(raw_g, v_mask_u8);
        raw_b = _mm256_and_si256(raw_b, v_mask_u8);

        // C. Convert to Float32
        __m256 f_r = _mm256_cvtepi32_ps(raw_r);
        __m256 f_g = _mm256_cvtepi32_ps(raw_g);
        __m256 f_b = _mm256_cvtepi32_ps(raw_b);

        // D. Normalize: (val * scale - mean) * inv_std
        // 1. Scale [0, 255] -> [0, 1]
        f_r = _mm256_mul_ps(f_r, v_scale);
        f_g = _mm256_mul_ps(f_g, v_scale);
        f_b = _mm256_mul_ps(f_b, v_scale);

        // 2. Subtract Mean
        f_r = _mm256_sub_ps(f_r, v_mean_r);
        f_g = _mm256_sub_ps(f_g, v_mean_g);
        f_b = _mm256_sub_ps(f_b, v_mean_b);

        // 3. Multiply Inverse Std
        f_r = _mm256_mul_ps(f_r, v_inv_std_r);
        f_g = _mm256_mul_ps(f_g, v_inv_std_g);
        f_b = _mm256_mul_ps(f_b, v_inv_std_b);

        // E. Store to NCHW planes
        _mm256_storeu_ps(dst_r + i, f_r);
        _mm256_storeu_ps(dst_g + i, f_g);
        _mm256_storeu_ps(dst_b + i, f_b);
    }

    // 3. Cleanup Loop (Scalar Fallback)
    // -------------------------------------------------------------------------
    for (; i < count; ++i) {
        const uint8_t* p = src + i * 3;
        dst[i]             = (p[0] * m_config.norm_params.scale_factor - m_config.norm_params.mean[0]) / m_config.norm_params.std[0];
        dst[count + i]     = (p[1] * m_config.norm_params.scale_factor - m_config.norm_params.mean[1]) / m_config.norm_params.std[1];
        dst[count * 2 + i] = (p[2] * m_config.norm_params.scale_factor - m_config.norm_params.mean[2]) / m_config.norm_params.std[2];
    }
}

void AvxImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    if (!src.data) return;

    // 1. Resize (Use OpenCV)
    // OpenCV's resize is already extremely optimized (AVX2/SSE4).
    cv::Mat img_wrapper(src.height, src.width, CV_8UC3, src.data);
    cv::Mat resized;

    if (src.width != m_config.target_width || src.height != m_config.target_height) {
        cv::resize(img_wrapper, resized, cv::Size(m_config.target_width, m_config.target_height), 0, 0, cv::INTER_LINEAR);
    } else {
        resized = img_wrapper;
    }

    // 2. Color Conversion
    // Our kernel expects RGB. If BGR, swap.
    if (src.format == ImageFormat::BGR && m_config.target_format == ImageFormat::RGB) {
        cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);
    }

    // 3. Normalize + Permute (Fused AVX Kernel)
    int total_pixels = m_config.target_width * m_config.target_height;
    float* dst_ptr = static_cast<float*>(dst.data());

    if (m_config.layout_nchw) {
        avx_normalize_permute(resized.data, dst_ptr, total_pixels);
    } else {
        // Fallback for NHWC (Standard copy)
        // TODO: Add AVX NHWC optimization
        float* d = dst_ptr;
        const uint8_t* s = resized.data;
        for (int k = 0; k < total_pixels * 3; ++k) {
            d[k] = s[k] * m_config.norm_params.scale_factor; // Simplified normalization
        }
    }
}

#else

// =================================================================================
// Non-AVX Fallback (Compilation safety)
// =================================================================================

void AvxImagePreprocessor::avx_normalize_permute(const uint8_t* src, float* dst, int count) {
    XINFER_LOG_ERROR("AVX2 instructions not supported by this build.");
}

void AvxImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    XINFER_LOG_ERROR("AvxImagePreprocessor called on non-AVX architecture.");
}

#endif

} // namespace xinfer::preproc