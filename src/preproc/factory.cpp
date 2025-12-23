#include <xinfer/preproc/factory.h>
#include <xinfer/core/logging.h>

// Universal Fallbacks
#include "image/cpu/opencv_image.h"
#include "audio/cpu/cpu_audio.h"

// Conditional Headers
#ifdef XINFER_HAS_CUDA
#include "image/cuda/cuda_image.h"
#endif

#ifdef XINFER_HAS_RGA
#include "image/rga/rga_image.h"
#endif

namespace xinfer::preproc {

    std::unique_ptr<IImagePreprocessor> create_image_preprocessor(Target target) {
        // 1. NVIDIA GPU -> Use CUDA
#ifdef XINFER_HAS_CUDA
        if (target == Target::NVIDIA_TRT) {
            XINFER_LOG_INFO("Using CUDA Preprocessor");
            return std::make_unique<CudaImagePreprocessor>();
        }
#endif

        // 2. Rockchip NPU -> Use RGA (Hardware 2D Engine)
#ifdef XINFER_HAS_RGA
        if (target == Target::ROCKCHIP_RKNN) {
            XINFER_LOG_INFO("Using Rockchip RGA Preprocessor");
            return std::make_unique<RgaImagePreprocessor>();
        }
#endif

        // 3. ARM Mobile -> Use NEON (TODO: Add NeonImagePreprocessor)
        // #if defined(__aarch64__)
        //     return std::make_unique<NeonImagePreprocessor>();
        // #endif

        // 4. Default -> OpenCV (CPU)
        XINFER_LOG_INFO("Using CPU (OpenCV) Preprocessor");
        return std::make_unique<OpenCVImagePreprocessor>();
    }

std::unique_ptr<IAudioPreprocessor> create_audio_preprocessor(Target target) {
    
    // 1. NVIDIA -> cuFFT
    #ifdef XINFER_ENABLE_CUDA
    if (target == Target::NVIDIA_TRT) {
        return std::make_unique<CuFFTAudioPreproc>();
    }
    #endif

    // 2. Apple -> Accelerate
    #ifdef __APPLE__
    if (target == Target::APPLE_COREML) {
        return std::make_unique<AccelerateAudioPreproc>();
    }
    #endif

    // 3. ARM (Rockchip, Pi, Jetson CPU) -> NEON
    #if defined(__aarch64__) || defined(__arm__)
    {
        auto ptr = std::make_unique<NeonAudioPreprocessor>();
        // Check if config can be satisfied by NEON?
        return ptr;
    }
    #endif

    // 4. Generic Fallback (Intel/AMD CPU) -> KissFFT
    return std::make_unique<CpuAudioPreprocessor>();
}

    std::unique_ptr<ITextPreprocessor> create_text_preprocessor(text::TokenizerType type) {
        switch (type) {
            case text::TokenizerType::BERT_WORDPIECE:
                return std::make_unique<text::BertTokenizer>();

            case text::TokenizerType::GPT_BPE:
                return std::make_unique<text::BpeTokenizer>();

            default:
                return std::make_unique<text::BertTokenizer>();
        }
    }

} // namespace xinfer::preproc