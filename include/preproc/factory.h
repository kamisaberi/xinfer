#pragma once

#include <memory>

// Include the Interfaces
#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/audio/audio_preprocessor.h>

// Include the Target definition
// (Ideally, move 'Target' enum to xinfer/core/types.h so Preproc doesn't depend on Compiler)
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::preproc {

    /**
     * @brief Factory for Image Preprocessors.
     *
     * Automatically selects the fastest implementation based on the target hardware.
     *
     * Strategy:
     * - If Target == NVIDIA_TRT -> Returns CudaImagePreprocessor
     * - If Target == ROCKCHIP_RKNN -> Returns RgaImagePreprocessor
     * - If Target == APPLE_COREML -> Returns MetalImagePreprocessor
     * - Else -> Returns OpenCVImagePreprocessor (CPU)
     *
     * @param target The hardware platform execution is running on.
     * @return std::unique_ptr<IImagePreprocessor>
     */
    std::unique_ptr<IImagePreprocessor> create_image_preprocessor(xinfer::Target target);

    /**
     * @brief Factory for Audio Preprocessors.
     *
     * Strategy:
     * - If Target == NVIDIA_TRT (and Batch > Threshold) -> Returns CuFFTAudioPreprocessor
     * - If Target == APPLE_COREML -> Returns AccelerateAudioPreprocessor
     * - Else -> Returns CpuAudioPreprocessor (Generic C++ / NEON)
     *
     * @param target The hardware platform execution is running on.
     * @return std::unique_ptr<IAudioPreprocessor>
     */
    std::unique_ptr<IAudioPreprocessor> create_audio_preprocessor(xinfer::Target target);

} // namespace xinfer::preproc