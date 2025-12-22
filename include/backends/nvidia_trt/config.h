#pragma once

#include <string>
#include <vector>
#include "types.h"

namespace xinfer::backends::nvidia {

/**
 * @brief Configuration for NVIDIA TensorRT Backend
 */
struct TrtConfig {
    // Path to the serialized TensorRT Engine (.engine / .plan)
    std::string model_path;

    // GPU Device Index (e.g., 0, 1)
    int device_id = 0;

    // CUDA Stream to use for inference.
    // If nullptr, xInfer creates its own internal stream.
    // Pass a raw `cudaStream_t` here to synchronize with external CUDA work.
    void* external_stream = nullptr;

    // Use DLA (Deep Learning Accelerator) on Jetson?
    DlaCore dla_core = DlaCore::GPU_FALLBACK;

    // Enable CUDA Graph capture for reduced CPU launch overhead?
    // Highly recommended for small models (Aegis Sky).
    bool enable_cuda_graphs = false;

    // Max workspace size in bytes (only relevant if JIT compiling, 
    // but mostly deprecated in TRT 10 in favor of internal pools).
    size_t workspace_size = 1024 * 1024 * 1024; // 1GB
};

} // namespace xinfer::backends::nvidia