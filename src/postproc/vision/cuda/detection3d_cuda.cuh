#pragma once

#include <xinfer/postproc/vision/detection3d_interface.h>
#include <cuda_runtime.h>
#include <vector>

namespace xinfer::postproc {

// Maximum number of candidates to keep on GPU before NMS.
// Keeping this reasonable prevents OOM and keeps nms fast.
constexpr int MAX_3D_CANDIDATES = 4096;

// Internal struct for GPU layout (aligned for memory coalescing)
struct alignas(16) GpuBox3D {
    float x, y, z;
    float w, l, h;
    float yaw;
    float velocity_x;
    float velocity_y;
    float score;
    float class_id;
};

/**
 * @brief CUDA Implementation of 3D Detection Post-processing.
 * 
 * Target: CenterPoint / PointPillars on Jetson Orin / dGPU.
 * 
 * Workflow:
 * 1. Kernel: Decode Heatmap + Reg map -> Compact Candidate Buffer (VRAM).
 * 2. Memcpy: Copy valid candidates to Host (Pinned Memory).
 * 3. CPU: Perform BEV NMS (Sorting/Filtering).
 */
class CudaDetection3DPostproc : public IDetection3DPostprocessor {
public:
    CudaDetection3DPostproc();
    ~CudaDetection3DPostproc() override;

    void init(const Detection3DConfig& config) override;

    std::vector<BoundingBox3D> process(const std::vector<core::Tensor>& tensors) override;

private:
    Detection3DConfig m_config;
    cudaStream_t m_stream = nullptr;

    // --- GPU Memory ---
    int* d_count = nullptr;       // Atomic counter
    GpuBox3D* d_candidates = nullptr; // Result buffer

    // --- Host Memory (Pinned) ---
    int* h_count = nullptr;
    GpuBox3D* h_candidates = nullptr;

    // Cached pre-calculated constants
    float m_min_x, m_min_y;
    float m_voxel_x, m_voxel_y;

    void allocate_buffers();
    void free_buffers();

    // Helper for CPU-side NMS
    void apply_nms_bev(std::vector<BoundingBox3D>& boxes);
};

} // namespace xinfer::postproc