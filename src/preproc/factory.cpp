
#include <xinfer/preproc/factory.h>
#include <xinfer/core/logging.h>

// --- Image Headers ---
#include "image/cpu/opencv_image.h"
#ifdef XINFER_HAS_CUDA
  #include "image/cuda/cuda_image.h"
#endif
#ifdef XINFER_HAS_RGA
  #include "image/rga/rga_image.h"
#endif
#ifdef __APPLE__
  #include "image/apple/metal_image.h"
#endif

// --- Audio Headers ---
#include "audio/cpu/cpu_audio.h"
#if defined(__aarch64__) || defined(__arm__)
  #include "audio/cpu/neon_audio.h"
#endif
#ifdef XINFER_HAS_CUDA
  #include "audio/cuda/cufft_audio.h"
#endif
#ifdef __APPLE__
  #include "audio/apple/accelerate_audio.h"
#endif

// --- Text Headers (NEW) ---
#include "text/cpu/bert_tokenizer.h"
#include "text/cpu/bpe_tokenizer.h"
#ifdef XINFER_HAS_CUDA
  #include "text/cuda/cubert_tokenizer.h"
#endif

// --- Tabular Headers (NEW) ---
#include "tabular/cpu/log_encoder.h"
#ifdef XINFER_HAS_CUDA
  #include "tabular/cuda/cudf_wrapper.h"
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

    std::unique_ptr<ITextPreprocessor> create_text_preprocessor(text::TokenizerType type, xinfer::Target target) {

        // 1. CUDA Acceleration (High Throughput / Batching)
#ifdef XINFER_HAS_CUDA
        if (target == xinfer::Target::NVIDIA_TRT) {
            // Only cuBERT (WordPiece) is currently implemented for GPU
            if (type == text::TokenizerType::BERT_WORDPIECE) {
                XINFER_LOG_INFO("Using CUDA Text Preprocessor (cuBERT)");
                return std::make_unique<text::CuBertTokenizer>();
            }
            XINFER_LOG_WARN("Requested GPU Tokenizer but type is not BERT. Falling back to CPU.");
        }
#endif

        // 2. CPU Fallback (Standard)
        switch (type) {
            case text::TokenizerType::BERT_WORDPIECE:
                return std::make_unique<text::BertTokenizer>();
            case text::TokenizerType::GPT_BPE:
            case text::TokenizerType::SENTENCEPIECE:
                return std::make_unique<text::BpeTokenizer>();
            default:
                XINFER_LOG_WARN("Unknown tokenizer type. Defaulting to BERT.");
                return std::make_unique<text::BertTokenizer>();
        }
    }

    std::unique_ptr<ITabularPreprocessor> create_tabular_preprocessor(xinfer::Target target) {

        // 1. CUDA Acceleration (cuDF / RAPIDS)
#ifdef XINFER_HAS_CUDA
        if (target == xinfer::Target::NVIDIA_TRT) {
            XINFER_LOG_INFO("Using CUDA Tabular Preprocessor (cuDF)");
            return std::make_unique<CudfTabularPreprocessor>();
        }
#endif

        // 2. CPU Fallback (Optimized C++)
        // This uses the log_encoder.cpp we wrote
        return std::make_unique<LogEncoder>();
    }

} // namespace xinfer::preproc