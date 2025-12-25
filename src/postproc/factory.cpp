#include <xinfer/postproc/factory.h>
#include <xinfer/core/logging.h>

// ============================================================================
// CPU Implementations (Universal Fallback)
// ============================================================================
#include "vision/cpu/yolo_postproc.h"
#include "vision/cpu/segmentation_cpu.h"
#include "vision/cpu/instance_seg_cpu.h"
#include "vision/cpu/anomaly_cpu.h"
#include "vision/cpu/detection3d_cpu.h"
#include "vision/cpu/classification_cpu.h"
#include "vision/cpu/tracker.h"

#include "text/cpu/ctc_decoder.h"
#include "text/cpu/llm_sampler.h"

#include "generative/cpu/diffusion_sampler.h"

// ============================================================================
// CUDA Implementations (NVIDIA Only)
// ============================================================================
#ifdef XINFER_HAS_CUDA
  #include "vision/cuda/yolo_postproc.cuh"
  #include "vision/cuda/segmentation_cuda.cuh"
  #include "vision/cuda/instance_seg_cuda.cuh"
  #include "vision/cuda/anomaly_cuda.cuh"
  #include "vision/cuda/detection3d_cuda.cuh"

  #include "text/cuda/ctc_decoder.cuh"

  #include "generative/cuda/diffusion_sampler.cuh"
#endif

// ============================================================================
// Apple Metal Implementations (Mac/iOS Only)
// ============================================================================
#ifdef __APPLE__
  // #include "vision/apple/metal_nms.h" // NMS logic is usually embedded in specific detectors
#endif

namespace xinfer::postproc {

// --- Vision Factory ---

std::unique_ptr<IDetectionPostprocessor> create_detection(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA Detection Post-processor");
        return std::make_unique<CudaYoloPostproc>();
    }
    #endif
    return std::make_unique<CpuYoloPostproc>();
}

std::unique_ptr<ISegmentationPostprocessor> create_segmentation(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA Segmentation Post-processor");
        return std::make_unique<CudaSegmentationPostproc>();
    }
    #endif
    return std::make_unique<CpuSegmentationPostproc>();
}

std::unique_ptr<IInstanceSegmentationPostprocessor> create_instance_segmentation(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA Instance Segmentation Post-processor");
        return std::make_unique<CudaInstanceSegPostproc>();
    }
    #endif
    return std::make_unique<CpuInstanceSegPostproc>();
}

std::unique_ptr<IAnomalyPostprocessor> create_anomaly(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA Anomaly Post-processor");
        return std::make_unique<CudaAnomalyPostproc>();
    }
    #endif
    return std::make_unique<CpuAnomalyPostproc>();
}

std::unique_ptr<IDetection3DPostprocessor> create_detection3d(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA 3D Detection Post-processor");
        return std::make_unique<CudaDetection3DPostproc>();
    }
    #endif
    return std::make_unique<CpuDetection3DPostproc>();
}

std::unique_ptr<IClassificationPostprocessor> create_classification(xinfer::Target target) {
    // Classification post-proc (Softmax + TopK) is usually fast enough on CPU.
    // CUDA version skipped for simplicity, but could be added for massive batches.
    return std::make_unique<CpuClassificationPostproc>();
}

std::unique_ptr<ITracker> create_tracker(xinfer::Target target) {
    // Tracking (Kalman Filter + Hungarian/Greedy Match) is sequential logic.
    // It runs best on CPU.
    return std::make_unique<CpuTracker>();
}

// --- Text Factory ---

std::unique_ptr<IOcrPostprocessor> create_ocr(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA CTC Decoder");
        return std::make_unique<CudaCtcDecoder>();
    }
    #endif
    return std::make_unique<CtcDecoder>();
}

std::unique_ptr<ILlmSampler> create_llm_sampler(xinfer::Target target) {
    // LLM Sampling is scalar math on small vectors (Vocab size).
    // CPU is standard for this step.
    return std::make_unique<CpuLlmSampler>();
}

// --- Generative Factory ---

std::unique_ptr<ISamplerPostprocessor> create_sampler(xinfer::Target target) {
    #ifdef XINFER_HAS_CUDA
    if (target == xinfer::Target::NVIDIA_TRT) {
        XINFER_LOG_INFO("Using CUDA Diffusion Sampler");
        return std::make_unique<CudaDiffusionSampler>();
    }
    #endif
    return std::make_unique<CpuDiffusionSampler>();
}

} // namespace xinfer::postproc