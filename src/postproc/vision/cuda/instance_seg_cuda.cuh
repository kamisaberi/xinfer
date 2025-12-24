#pragma once

#include <xinfer/postproc/vision/instance_seg_interface.h>
#include <cuda_runtime.h>
#include <vector>

namespace xinfer::postproc {

constexpr int MAX_SEG_CANDIDATES = 4096;
constexpr int MASK_PROTO_CHANNELS = 32;
constexpr int PROTO_H = 160;
constexpr int PROTO_W = 160;

// Aligned structure for GPU access
struct alignas(16) GpuSegCandidate {
    float x1, y1, x2, y2;
    float score;
    float class_id;
    float mask_coeffs[MASK_PROTO_CHANNELS]; // 32 coeffs
};

/**
 * @brief CUDA Implementation of Instance Segmentation.
 * 
 * Pipeline:
 * 1. Decode Kernel: Extracts boxes and mask coefficients.
 * 2. CPU NMS: Filters boxes (faster on CPU for low counts).
 * 3. Mask Kernel: Computes (Coeffs * Protos), performs Sigmoid, 
 *    Upsamples to image size, Crops to box, and Thresholds.
 */
class CudaInstanceSegPostproc : public IInstanceSegmentationPostprocessor {
public:
    CudaInstanceSegPostproc();
    ~CudaInstanceSegPostproc() override;

    void init(const InstanceSegConfig& config) override;

    std::vector<InstanceResult> process(const std::vector<core::Tensor>& tensors) override;

private:
    InstanceSegConfig m_config;
    cudaStream_t m_stream = nullptr;

    // --- GPU Buffers ---
    int* d_count = nullptr;
    GpuSegCandidate* d_candidates = nullptr; // Raw detections before NMS
    
    // Buffers for Post-NMS processing
    // We copy valid boxes back to GPU to generate masks
    GpuSegCandidate* d_final_dets = nullptr;
    uint8_t* d_final_masks = nullptr; // [MaxDets, TargetH, TargetW]

    // --- Host Buffers (Pinned) ---
    int* h_count = nullptr;
    GpuSegCandidate* h_candidates = nullptr;

    void allocate_buffers();
    void free_buffers();
};

} // namespace xinfer::postproc