#include <include/postproc/detection3d.h>
#include <cuda_runtime.h>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::postproc::detection3d {

// --- CUDA KERNEL FOR 3D IOU CALCULATION ---
// This is a simplified version focusing on Birds-Eye-View (BEV) IoU, which is common.
// A full 3D IoU is much more complex.
__device__ inline float bev_iou(const float* box1, const float* box2) {
    // box format: [cx, cy, cz, l, w, h, yaw]
    float area1 = box1[3] * box1[4];
    float area2 = box2[3] * box2[4];

    // Simplified intersection approximation
    float dx = fabs(box1[0] - box2[0]);
    float dy = fabs(box1[1] - box2[1]);
    float inter_l = fmaxf(0.0f, (box1[3] + box2[3]) / 2.0f - dx);
    float inter_w = fmaxf(0.0f, (box1[4] + box2[4]) / 2.0f - dy);
    float inter_area = inter_l * inter_w;

    float union_area = area1 + area2 - inter_area;
    return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
}

__global__ void nms_3d_kernel(
    const float* boxes,         // [N, 7]
    const int* sorted_indices,  // [N]
    unsigned long long* suppressed_mask, // Bitmask for suppressed boxes
    int num_boxes,
    float iou_threshold)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_boxes) return;

    int current_box_idx = sorted_indices[idx];

    // Check if this box has already been suppressed by a higher-scoring one
    if ((suppressed_mask[current_box_idx / 64] >> (current_box_idx % 64)) & 1ULL) {
        return;
    }

    const float* current_box = boxes + current_box_idx * 7;

    // Iterate through all lower-scoring boxes
    for (int i = idx + 1; i < num_boxes; ++i) {
        int other_box_idx = sorted_indices[i];
        if ((suppressed_mask[other_box_idx / 64] >> (other_box_idx % 64)) & 1ULL) {
            continue;
        }

        const float* other_box = boxes + other_box_idx * 7;
        float iou = bev_iou(current_box, other_box);

        if (iou > iou_threshold) {
            // Suppress the lower-scoring box
            atomicOr(&suppressed_mask[other_box_idx / 64], 1ULL << (other_box_idx % 64));
        }
    }
}


std::vector<int> nms(
    const core::Tensor& decoded_boxes,
    const core::Tensor& decoded_scores,
    float nms_iou_threshold)
{
    auto box_shape = decoded_boxes.shape();
    if (box_shape.size() != 2 || box_shape[1] != 7) {
        throw std::invalid_argument("3D NMS boxes tensor must have shape [N, 7]");
    }
    const int num_boxes = box_shape[0];
    if (num_boxes == 0) return {};

    // --- Step 1: Sort boxes by score in descending order (on GPU) ---
    // Use Thrust for efficient parallel sorting.
    thrust::device_ptr<const float> d_scores_ptr((const float*)decoded_scores.data());
    thrust::device_vector<int> d_indices(num_boxes);
    thrust::sequence(d_indices.begin(), d_indices.end());

    thrust::sort_by_key(thrust::device, d_scores_ptr, d_scores_ptr + num_boxes, d_indices.begin(), thrust::greater<float>());

    // --- Step 2: Prepare suppression mask on GPU ---
    int mask_size = (num_boxes + 63) / 64;
    unsigned long long* d_suppressed_mask;
    CHECK_CUDA(cudaMalloc(&d_suppressed_mask, mask_size * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_suppressed_mask, 0, mask_size * sizeof(unsigned long long)));

    // --- Step 3: Launch the NMS kernel ---
    int threads = 256;
    int blocks = (num_boxes + threads - 1) / threads;
    nms_3d_kernel<<<blocks, threads>>>(
        (const float*)decoded_boxes.data(),
        thrust::raw_pointer_cast(d_indices.data()),
        d_suppressed_mask,
        num_boxes,
        nms_iou_threshold
    );
    CHECK_CUDA(cudaGetLastError());

    // --- Step 4: Copy results back to CPU ---
    std::vector<unsigned long long> h_suppressed_mask(mask_size);
    CHECK_CUDA(cudaMemcpy(h_suppressed_mask.data(), d_suppressed_mask, mask_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    std::vector<int> h_indices(num_boxes);
    CHECK_CUDA(cudaMemcpy(h_indices.data(), thrust::raw_pointer_cast(d_indices.data()), num_boxes * sizeof(int), cudaMemcpyDeviceToHost));

    // --- Step 5: Collect the indices of non-suppressed boxes ---
    std::vector<int> final_indices;
    for (int i = 0; i < num_boxes; ++i) {
        int box_idx = h_indices[i];
        if (!((h_suppressed_mask[box_idx / 64] >> (box_idx % 64)) & 1ULL)) {
            final_indices.push_back(box_idx);
        }
    }

    CHECK_CUDA(cudaFree(d_suppressed_mask));
    return final_indices;
}

} // namespace xinfer::postproc::detection3d