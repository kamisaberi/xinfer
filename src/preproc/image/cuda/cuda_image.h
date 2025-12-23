#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>
#include <cuda_runtime.h>

namespace xinfer::preproc {

    /**
     * @brief CUDA Image Preprocessor
     *
     * Uses custom CUDA kernels to perform:
     * 1. Resize (Bilinear/Nearest)
     * 2. Color Space Conversion (BGR->RGB)
     * 3. Normalization ((x - mean) / std)
     * 4. Layout Change (HWC -> NCHW)
     *
     * All in a single kernel launch.
     */
    class CudaImagePreprocessor : public IImagePreprocessor {
    public:
        CudaImagePreprocessor();
        ~CudaImagePreprocessor() override;

        void init(const ImagePreprocConfig& config) override;

        /**
         * @brief Process image on GPU.
         *
         * If src.data is on CPU, it automatically copies it to an internal GPU buffer
         * before processing. For best performance, ensure src.data is already on GPU.
         */
        void process(const ImageFrame& src, core::Tensor& dst) override;

    private:
        ImagePreprocConfig m_config;

        // Internal scratch buffer used if input comes from CPU
        void* d_input_buffer = nullptr;
        size_t d_input_capacity = 0;

        // Helper to ensure internal buffer is large enough
        void reserve_input_buffer(size_t size_bytes);
    };

} // namespace xinfer::preproc