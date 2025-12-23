#include <xinfer/preproc/factory.h>

// Include headers conditionally based on CMake flags
#ifdef XINFER_ENABLE_CUDA
  #include "image/cuda/cuda_image.h"
  #include "audio/cuda/cufft_audio.h"
#endif

#ifdef XINFER_ENABLE_RKNN
  #include "image/rga/rga_image.h"
#endif

#ifdef __APPLE__
  #include "image/apple/metal_image.h"
  #include "audio/apple/accelerate_audio.h"
#endif

#include "image/cpu/opencv_image.h"
#include "audio/cpu/kissfft_audio.h"

namespace xinfer::preproc {

    std::unique_ptr<IImagePreprocessor> create_image_preprocessor(Target target) {
#ifdef XINFER_ENABLE_CUDA
        if (target == Target::NVIDIA_TRT) return std::make_unique<CudaImagePreproc>();
#endif

#ifdef XINFER_ENABLE_RKNN
        if (target == Target::ROCKCHIP_RKNN) return std::make_unique<RgaImagePreproc>();
#endif

#ifdef __APPLE__
        if (target == Target::APPLE_COREML) return std::make_unique<MetalImagePreproc>();
#endif

        // Default Fallback
        return std::make_unique<OpenCVImagePreproc>();
    }

    std::unique_ptr<IAudioPreprocessor> create_audio_preprocessor(Target target) {
#ifdef XINFER_ENABLE_CUDA
        // Only use GPU for audio if batch size is huge, otherwise CPU is faster due to latency
        if (target == Target::NVIDIA_TRT) return std::make_unique<CuFFTAudioPreproc>();
#endif

#ifdef __APPLE__
        return std::make_unique<AccelerateAudioPreproc>();
#endif

        // Standard CPU Fallback (KissFFT or FFTW)
        return std::make_unique<CpuAudioPreproc>();
    }

}