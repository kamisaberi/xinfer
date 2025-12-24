#include "instance_seg_cuda.cuh"
#include <xinfer/core/logging.h>
#include <opencv2/dnn.hpp> // For CPU NMS
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// Kernel 1: Decode Boxes & Coeffs
// =================================================================================
__global__ void decode_seg_kernel(const float* __restrict__ det_data,
                                  int* __restrict__ count,
                                  GpuSegCandidate* __restrict__ candidates,
                                  int num_classes,
                                  int num_anchors,
                                  int num_masks,
                                  float threshold,
                                  int max_candidates) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    // Pointers (YOLOv8 layout: [Batch, Channels, Anchors])
    // Channels = 4 (Box) + NumClasses + 32 (Masks)
    const float* p_cls = det_data + (4 * num_anchors);
    const float* p_masks = det_data + ((4 + num_classes) * num_anchors);

    // Find Max Score
    float max_score = 0.0f;
    int class_id = -1;

    for (int c = 0; c < num_classes; ++c) {
        float score = p_cls[c * num_anchors + idx];
        if (score > max_score) {
            max_score = score;
            class_id = c;
        }
    }

    if (max_score >= threshold) {
        int write_idx = atomicAdd(count, 1);
        if (write_idx < max_candidates) {
            // Decode Box
            float cx = det_data[0 * num_anchors + idx];
            float cy = det_data[1 * num_anchors + idx];
            float w  = det_data[2 * num_anchors + idx];
            float h  = det_data[3 * num_anchors + idx];

            GpuSegCandidate cand;
            cand.x1 = cx - w * 0.5f;
            cand.y1 = cy - h * 0.5f;
            cand.x2 = cx + w * 0.5f;
            cand.y2 = cy + h * 0.5f;
            cand.score = max_score;
            cand.class_id = (float)class_id;

            // Copy Mask Coefficients
            for (int k = 0; k < num_masks; ++k) {
                cand.mask_coeffs[k] = p_masks[k * num_anchors + idx];
            }

            candidates[write_idx] = cand;
        }
    }
}

// =================================================================================
// Kernel 2: Assemble, Upsample, Crop, Threshold Masks
// =================================================================================
// Grid: [TargetW, TargetH, NumDetections]
// Each thread computes one pixel of the final mask for one detection.
__global__ void assemble_masks_kernel(const GpuSegCandidate* __restrict__ detections,
                                      const float* __restrict__ protos, // [32, 160, 160]
                                      uint8_t* __restrict__ output_masks,
                                      int num_detections,
                                      int num_masks,
                                      int proto_h, int proto_w,
                                      int target_h, int target_w) 
{
    // Indices
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Target X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Target Y
    int d = blockIdx.z;                            // Detection Index

    if (x >= target_w || y >= target_h || d >= num_detections) return;

    // 1. Check Crop (Is this pixel inside the bounding box?)
    // Optimization: Don't compute mask logic for background pixels
    const GpuSegCandidate& det = detections[d];
    if (x < det.x1 || x > det.x2 || y < det.y1 || y > det.y2) {
        // Outside box -> 0
        output_masks[d * (target_w * target_h) + y * target_w + x] = 0;
        return;
    }

    // 2. Map Target(x,y) -> Proto(px, py) (Bilinear Interpolation coords)
    // Scale factor
    float scale_x = (float)proto_w / target_w;
    float scale_y = (float)proto_h / target_h;
    
    float px = x * scale_x;
    float py = y * scale_y;

    // Bilinear weights
    int x_low = (int)px;
    int y_low = (int)py;
    int x_high = min(x_low + 1, proto_w - 1);
    int y_high = min(y_low + 1, proto_h - 1);

    float fx = px - x_low;
    float fy = py - y_low;
    float w_tl = (1.0f - fx) * (1.0f - fy);
    float w_tr = fx * (1.0f - fy);
    float w_bl = (1.0f - fx) * fy;
    float w_br = fx * fy;

    // 3. Compute Dot Product (Linear Combination of Prototypes)
    float sum_val = 0.0f;
    int plane_size = proto_h * proto_w;

    for (int k = 0; k < num_masks; ++k) {
        float coeff = det.mask_coeffs[k];
        
        // Fetch 4 neighbors from proto k
        // Protos are usually [32, 160, 160] (Planar)
        const float* plane = protos + (k * plane_size);
        float v_tl = plane[y_low * proto_w + x_low];
        float v_tr = plane[y_low * proto_w + x_high];
        float v_bl = plane[y_high * proto_w + x_low];
        float v_br = plane[y_high * proto_w + x_high];

        float interpolated = w_tl * v_tl + w_tr * v_tr + w_bl * v_bl + w_br * v_br;
        sum_val += coeff * interpolated;
    }

    // 4. Sigmoid Activation
    float prob = 1.0f / (1.0f + expf(-sum_val));

    // 5. Threshold
    output_masks[d * (target_w * target_h) + y * target_w + x] = (prob > 0.5f) ? 255 : 0;
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaInstanceSegPostproc::CudaInstanceSegPostproc() {
    allocate_buffers();
    cudaStreamCreate(&m_stream);
}

CudaInstanceSegPostproc::~CudaInstanceSegPostproc() {
    free_buffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaInstanceSegPostproc::allocate_buffers() {
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_candidates, MAX_SEG_CANDIDATES * sizeof(GpuSegCandidate));
    
    // Allocate space for final detections after NMS to be sent back to GPU
    cudaMalloc(&d_final_dets, m_config.max_detections * sizeof(GpuSegCandidate));
    
    // Mask Output: [MaxDets, H, W] bytes
    size_t mask_size = m_config.max_detections * m_config.target_height * m_config.target_width;
    cudaMalloc(&d_final_masks, mask_size);

    cudaMallocHost(&h_count, sizeof(int));
    cudaMallocHost(&h_candidates, MAX_SEG_CANDIDATES * sizeof(GpuSegCandidate));
}

void CudaInstanceSegPostproc::free_buffers() {
    if (d_count) cudaFree(d_count);
    if (d_candidates) cudaFree(d_candidates);
    if (d_final_dets) cudaFree(d_final_dets);
    if (d_final_masks) cudaFree(d_final_masks);
    if (h_count) cudaFreeHost(h_count);
    if (h_candidates) cudaFreeHost(h_candidates);
}

void CudaInstanceSegPostproc::init(const InstanceSegConfig& config) {
    m_config = config;
    // Reallocate if config changes max_detections or resolution? 
    // For simplicity assuming fixed config after init.
}

std::vector<InstanceResult> CudaInstanceSegPostproc::process(const std::vector<core::Tensor>& tensors) {
    if (tensors.size() < 2) return {};

    const auto& det_tensor = tensors[0];
    const auto& proto_tensor = tensors[1];

    int num_anchors = (int)det_tensor.shape()[2];
    int num_masks = m_config.num_mask_protos; // 32

    // 1. Decode & Filter (GPU)
    cudaMemsetAsync(d_count, 0, sizeof(int), m_stream);
    
    int threads = 256;
    int blocks = (num_anchors + threads - 1) / threads;

    decode_seg_kernel<<<blocks, threads, 0, m_stream>>>(
        static_cast<const float*>(det_tensor.data()),
        d_count,
        d_candidates,
        m_config.num_classes,
        num_anchors,
        num_masks,
        m_config.conf_threshold,
        MAX_SEG_CANDIDATES
    );

    // 2. Download Candidates (D2H)
    cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    int count = *h_count;
    if (count > MAX_SEG_CANDIDATES) count = MAX_SEG_CANDIDATES;
    if (count == 0) return {};

    cudaMemcpyAsync(h_candidates, d_candidates, count * sizeof(GpuSegCandidate), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 3. CPU NMS
    std::vector<cv::Rect> boxes_cv;
    std::vector<float> scores_cv;
    for (int i = 0; i < count; ++i) {
        const auto& c = h_candidates[i];
        boxes_cv.emplace_back((int)c.x1, (int)c.y1, (int)(c.x2 - c.x1), (int)(c.y2 - c.y1));
        scores_cv.push_back(c.score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_cv, scores_cv, m_config.conf_threshold, m_config.nms_threshold, indices);

    if (indices.size() > (size_t)m_config.max_detections) indices.resize(m_config.max_detections);
    int final_count = indices.size();

    // 4. Upload Valid Detections (H2D)
    // We need them on GPU to compute masks
    std::vector<GpuSegCandidate> final_dets_host;
    final_dets_host.reserve(final_count);
    
    std::vector<InstanceResult> results;
    results.reserve(final_count);

    for (int idx : indices) {
        final_dets_host.push_back(h_candidates[idx]);
        
        // Prepare result struct (CPU side)
        InstanceResult res;
        res.box = {
            h_candidates[idx].x1, h_candidates[idx].y1, 
            h_candidates[idx].x2, h_candidates[idx].y2, 
            h_candidates[idx].score, (int)h_candidates[idx].class_id
        };
        res.mask.resize({1, (int64_t)m_config.target_height, (int64_t)m_config.target_width}, core::DataType::kUINT8);
        results.push_back(res);
    }

    cudaMemcpyAsync(d_final_dets, final_dets_host.data(), final_count * sizeof(GpuSegCandidate), cudaMemcpyHostToDevice, m_stream);

    // 5. Generate Masks (GPU)
    dim3 mask_block(16, 16);
    dim3 mask_grid(
        (m_config.target_width + mask_block.x - 1) / mask_block.x,
        (m_config.target_height + mask_block.y - 1) / mask_block.y,
        final_count
    );

    assemble_masks_kernel<<<mask_grid, mask_block, 0, m_stream>>>(
        d_final_dets,
        static_cast<const float*>(proto_tensor.data()),
        d_final_masks,
        final_count,
        num_masks,
        PROTO_H, PROTO_W,
        m_config.target_height, m_config.target_width
    );

    // 6. Download Masks (D2H)
    // We download all masks in one block to a temp buffer or directly to results
    size_t mask_area = m_config.target_height * m_config.target_width;
    std::vector<uint8_t> all_masks(final_count * mask_area);
    
    cudaMemcpyAsync(all_masks.data(), d_final_masks, all_masks.size(), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // Distribute to individual result structs
    for (int i = 0; i < final_count; ++i) {
        std::memcpy(results[i].mask.data(), 
                    all_masks.data() + (i * mask_area), 
                    mask_area);
    }

    return results;
}

} // namespace xinfer::postproc