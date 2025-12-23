#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>

namespace xinfer::preproc {

    /**
     * @brief AVX2 Image Preprocessor
     *
     * Optimized for Intel/AMD x86_64 CPUs supporting AVX2.
     *
     * Performance Strategy:
     * 1. OpenCV for Geometric Resize (already highly optimized).
     * 2. Custom AVX2 Intrinsics for HWC -> NCHW + Normalization.
     *    - Uses _mm256_i32gather_epi32 to deinterleave RGB data.
     *    - Uses Fused Multiply-Add (FMA) for normalization.
     */
    class AvxImagePreprocessor : public IImagePreprocessor {
    public:
        AvxImagePreprocessor() = default;
        ~AvxImagePreprocessor() override = default;

        void init(const ImagePreprocConfig& config) override;
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
        ImagePreprocConfig m_config;

        /**
         * @brief Fused Kernel: Uint8 HWC -> Float32 NCHW
         * Processes 8 pixels per iteration using 256-bit registers.
         */
        void avx_normalize_permute(const uint8_t* src, float* dst, int count);
    };

} // namespace xinfer::preproc