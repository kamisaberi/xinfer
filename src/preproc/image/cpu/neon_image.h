#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>

namespace xinfer::preproc {

    /**
     * @brief ARM NEON Image Preprocessor
     *
     * Optimized for ARMv8/AArch64 architectures (Rockchip, RPi, Jetson, Apple M-series).
     *
     * Strategy:
     * 1. Uses OpenCV for Geometric Resize (OpenCV's NEON resize is hard to beat).
     * 2. Uses Custom NEON Intrinsics for Color Conv + Normalize + HWC->NCHW Permutation.
     *    (This step is where standard libraries perform poorly).
     */
    class NeonImagePreprocessor : public IImagePreprocessor {
    public:
        NeonImagePreprocessor() = default;
        ~NeonImagePreprocessor() override = default;

        void init(const ImagePreprocConfig& config) override;
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
        ImagePreprocConfig m_config;

        /**
         * @brief Fused NEON Kernel: Uint8 HWC -> Float32 NCHW + Normalize
         *
         * Processes 16 pixels at a time using 128-bit SIMD registers.
         */
        void neon_normalize_permute(const uint8_t* src, float* dst, int count);
    };

} // namespace xinfer::preproc