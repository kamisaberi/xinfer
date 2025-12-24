#pragma once

#include <xinfer/postproc/vision/detection_interface.h>
#include <cuda_runtime.h>
#include <vector>

namespace xinfer::postproc {

// Max number of raw candidates to keep on GPU before NMS
// Prevents buffer overflows in the kernel
constexpr int MAX_GPU_CANDIDATES = 4096;

// Internal struct for GPU processing
struct alignas(16) GpuBox {
    float x1, y1, x2, y2;
    float score;
    float class_id; // float to align structure, cast to int later
};

/**
 * @brief CUDA Implementation of YOLO Post-processing.
 * 
 * Offloads the heavy "Grid Walking" and "Box Decoding" to the GPU.
 * Uses atomic counters to filter low-confidence boxes directly in VRAM.
 */
class CudaYoloPostproc : public IDetectionPostprocessor {
public:
    CudaYoloPostproc();
    ~CudaYoloPostproc() override;

    void init(const DetectionConfig& config) override;

    std::vector<BoundingBox> process(const std::vector<core::Tensor>& tensors) override;

private:
    DetectionConfig m_config;

    // --- GPU Buffers ---
    // Stores [Count, Box0, Box1, ...]
    // Index 0 contains the atomic counter (number of detected boxes)
    int* d_count = nullptr;      
    GpuBox* d_candidates = nullptr;

    // Host buffer to receive candidates
    int* h_count = nullptr; // Pinned memory
    GpuBox* h_candidates = nullptr; // Pinned memory

    // Cached CUDA stream
    cudaStream_t m_stream = nullptr;

    void allocate_buffers();
    void free_buffers();
};

} // namespace xinfer::postproc