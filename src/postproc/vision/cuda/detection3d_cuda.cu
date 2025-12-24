#include "detection3d_cuda.cuh"
#include <xinfer/core/logging.h>
#include <cmath>
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernels
// =================================================================================

/**
 * @brief Decode CenterPoint/PointPillars Output
 * 
 * Inputs:
 * - heatmap: [Channels, H, W] (Flattened)
 * - regression: [RegChannels, H, W] (Flattened)
 * 
 * Logic:
 * Each thread processes one grid cell. If score > thresh, decode params and append.
 */
__global__ void decode_3d_kernel(const float* __restrict__ heatmap,
                                 const float* __restrict__ regression,
                                 int* __restrict__ count,
                                 GpuBox3D* __restrict__ candidates,
                                 int num_classes,
                                 int height,
                                 int width,
                                 int reg_channels,
                                 float score_thresh,
                                 float min_x, float min_y,
                                 float voxel_x, float voxel_y,
                                 int max_candidates) 
{
    // 1. Calculate Grid Position
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int area = height * width;
    
    if (idx >= area) return;

    int y = idx / width;
    int x = idx % width;

    // 2. Find Max Score across classes for this pixel
    // Heatmap layout: [Classes, H, W] -> Plane stride is 'area'
    float max_score = 0.0f;
    int class_id = -1;

    for (int c = 0; c < num_classes; ++c) {
        float s = heatmap[c * area + idx];
        if (s > max_score) {
            max_score = s;
            class_id = c;
        }
    }

    // 3. Threshold Check
    if (max_score > score_thresh) {
        // 4. Atomic Allocation
        int write_idx = atomicAdd(count, 1);

        if (write_idx < max_candidates) {
            // 5. Decode Regression
            // Reg layout: [RegChannels, H, W]
            // Standard CenterPoint offsets:
            // 0: dx, 1: dy, 2: z, 3: log_w, 4: log_l, 5: log_h, 6: sin, 7: cos, 8: vx, 9: vy
            
            float ox = regression[0 * area + idx];
            float oy = regression[1 * area + idx];
            float z  = regression[2 * area + idx];
            float dw = regression[3 * area + idx];
            float dl = regression[4 * area + idx];
            float dh = regression[5 * area + idx];
            float rot_s = regression[6 * area + idx];
            float rot_c = regression[7 * area + idx];

            // Velocity is optional (check reg_channels size in host code, assuming 10 here)
            float vx = 0.0f;
            float vy = 0.0f;
            if (reg_channels >= 10) {
                vx = regression[8 * area + idx];
                vy = regression[9 * area + idx];
            }

            // 6. Coordinate Transform
            GpuBox3D box;
            box.x = (x + ox) * voxel_x + min_x;
            box.y = (y + oy) * voxel_y + min_y;
            box.z = z;
            
            box.w = expf(dw);
            box.l = expf(dl);
            box.h = expf(dh);
            
            box.yaw = atan2f(rot_s, rot_c);
            
            box.velocity_x = vx;
            box.velocity_y = vy;
            box.score = max_score;
            box.class_id = (float)class_id;

            // Write to Global Memory
            candidates[write_idx] = box;
        }
    }
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaDetection3DPostproc::CudaDetection3DPostproc() {
    allocate_buffers();
    cudaStreamCreate(&m_stream);
}

CudaDetection3DPostproc::~CudaDetection3DPostproc() {
    free_buffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaDetection3DPostproc::allocate_buffers() {
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_candidates, MAX_3D_CANDIDATES * sizeof(GpuBox3D));
    
    cudaMallocHost(&h_count, sizeof(int));
    cudaMallocHost(&h_candidates, MAX_3D_CANDIDATES * sizeof(GpuBox3D));
}

void CudaDetection3DPostproc::free_buffers() {
    if (d_count) cudaFree(d_count);
    if (d_candidates) cudaFree(d_candidates);
    if (h_count) cudaFreeHost(h_count);
    if (h_candidates) cudaFreeHost(h_candidates);
}

void CudaDetection3DPostproc::init(const Detection3DConfig& config) {
    m_config = config;
    
    // Precalc constants
    m_min_x = m_config.pc_range[0];
    m_min_y = m_config.pc_range[1];
    m_voxel_x = m_config.voxel_size_x * m_config.downsample_ratio;
    m_voxel_y = m_config.voxel_size_y * m_config.downsample_ratio;
}

std::vector<BoundingBox3D> CudaDetection3DPostproc::process(const std::vector<core::Tensor>& tensors) {
    // 1. Validation
    if (tensors.size() < 2) {
        XINFER_LOG_ERROR("Detection3D expects Heatmap and Regression tensors.");
        return {};
    }

    const auto& hm_tensor = tensors[0];
    const auto& reg_tensor = tensors[1];

    int num_classes = (int)hm_tensor.shape()[1];
    int height = (int)hm_tensor.shape()[2];
    int width = (int)hm_tensor.shape()[3];
    int reg_channels = (int)reg_tensor.shape()[1];

    // Get Device Pointers (Assuming tensor data is on GPU)
    const float* d_hm = static_cast<const float*>(hm_tensor.data());
    const float* d_reg = static_cast<const float*>(reg_tensor.data());

    // 2. Reset Counter
    cudaMemsetAsync(d_count, 0, sizeof(int), m_stream);

    // 3. Launch Kernel
    int total_pixels = height * width;
    int threads = 256;
    int blocks = (total_pixels + threads - 1) / threads;

    decode_3d_kernel<<<blocks, threads, 0, m_stream>>>(
        d_hm, d_reg, d_count, d_candidates,
        num_classes, height, width, reg_channels,
        m_config.score_threshold,
        m_min_x, m_min_y, m_voxel_x, m_voxel_y,
        MAX_3D_CANDIDATES
    );

    // 4. Download Count
    cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    int count = *h_count;
    if (count > MAX_3D_CANDIDATES) count = MAX_3D_CANDIDATES;
    if (count == 0) return {};

    // 5. Download Candidates
    cudaMemcpyAsync(h_candidates, d_candidates, count * sizeof(GpuBox3D), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 6. CPU NMS
    // Convert GpuBox3D -> BoundingBox3D
    std::vector<BoundingBox3D> boxes;
    boxes.reserve(count);

    for (int i = 0; i < count; ++i) {
        const auto& gb = h_candidates[i];
        BoundingBox3D b;
        b.x = gb.x; b.y = gb.y; b.z = gb.z;
        b.w = gb.w; b.l = gb.l; b.h = gb.h;
        b.yaw = gb.yaw;
        b.velocity_x = gb.velocity_x;
        b.velocity_y = gb.velocity_y;
        b.score = gb.score;
        b.class_id = (int)gb.class_id;
        boxes.push_back(b);
    }

    apply_nms_bev(boxes);
    return boxes;
}

// 7. CPU NMS Implementation (BEV Distance Greedy)
void CudaDetection3DPostproc::apply_nms_bev(std::vector<BoundingBox3D>& boxes) {
    // Sort descending by score
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox3D& a, const BoundingBox3D& b) {
        return a.score > b.score;
    });

    std::vector<BoundingBox3D> picked;
    picked.reserve(std::min((size_t)m_config.max_detections, boxes.size()));
    
    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        picked.push_back(boxes[i]);
        if (picked.size() >= m_config.max_detections) break;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            // Only suppress same class
            if (boxes[i].class_id != boxes[j].class_id) continue;

            // Fast Distance Check (Approximate IoU for BEV)
            float dx = boxes[i].x - boxes[j].x;
            float dy = boxes[i].y - boxes[j].y;
            float dist_sq = dx*dx + dy*dy;
            
            // Derive threshold from object size (Radius check)
            float radius = (boxes[i].w + boxes[i].l) / 4.0f; 
            // 2D Distance threshold typically around 0.5m - 2.0m depending on IoU config
            float dist_thresh = radius * (1.0f - m_config.nms_threshold) * 2.0f; 

            if (dist_sq < (dist_thresh * dist_thresh)) {
                suppressed[j] = true;
            }
        }
    }
    boxes = std::move(picked);
}

} // namespace xinfer::postproc