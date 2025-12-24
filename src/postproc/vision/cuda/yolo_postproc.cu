#include "yolo_postproc.cuh"
#include <xinfer/core/logging.h>

#include <opencv2/dnn.hpp> // For final NMS
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernel
// =================================================================================

/**
 * @brief Decodes YOLOv8 Output (Anchor-Free)
 * Input Shape: [Batch, Channels, Anchors] -> e.g. [1, 84, 8400]
 * Layout: 
 *   Row 0: cx
 *   Row 1: cy
 *   Row 2: w
 *   Row 3: h
 *   Row 4..N: Class Scores
 */
__global__ void decode_yolov8_kernel(const float* __restrict__ data,
                                     int* __restrict__ count,
                                     GpuBox* __restrict__ candidates,
                                     int num_classes,
                                     int num_anchors,
                                     float threshold,
                                     int max_candidates) 
{
    // 1. Calculate Anchor Index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_anchors) return;

    // 2. Pointers to rows (Strided by num_anchors)
    const float* p_scores = data + (4 * num_anchors); // Start of class scores

    // 3. Find Max Score
    float max_score = 0.0f;
    int class_id = -1;

    // Iterate classes for this anchor
    // Optimization: Unroll small loops if num_classes is fixed, but dynamic here
    for (int c = 0; c < num_classes; ++c) {
        float score = p_scores[c * num_anchors + idx];
        if (score > max_score) {
            max_score = score;
            class_id = c;
        }
    }

    // 4. Threshold Check
    if (max_score >= threshold) {
        // Atomic Add to get index
        int write_idx = atomicAdd(count, 1);

        if (write_idx < max_candidates) {
            // Decode Box
            float cx = data[0 * num_anchors + idx];
            float cy = data[1 * num_anchors + idx];
            float w  = data[2 * num_anchors + idx];
            float h  = data[3 * num_anchors + idx];

            GpuBox box;
            box.x1 = cx - w * 0.5f;
            box.y1 = cy - h * 0.5f;
            box.x2 = cx + w * 0.5f;
            box.y2 = cy + h * 0.5f;
            box.score = max_score;
            box.class_id = (float)class_id;

            candidates[write_idx] = box;
        }
    }
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaYoloPostproc::CudaYoloPostproc() {
    allocate_buffers();
    cudaStreamCreate(&m_stream);
}

CudaYoloPostproc::~CudaYoloPostproc() {
    free_buffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaYoloPostproc::allocate_buffers() {
    // Device Alloc
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_candidates, MAX_GPU_CANDIDATES * sizeof(GpuBox));

    // Host Alloc (Pinned memory for faster D2H)
    cudaMallocHost(&h_count, sizeof(int));
    cudaMallocHost(&h_candidates, MAX_GPU_CANDIDATES * sizeof(GpuBox));
}

void CudaYoloPostproc::free_buffers() {
    if (d_count) cudaFree(d_count);
    if (d_candidates) cudaFree(d_candidates);
    if (h_count) cudaFreeHost(h_count);
    if (h_candidates) cudaFreeHost(h_candidates);
}

void CudaYoloPostproc::init(const DetectionConfig& config) {
    m_config = config;
}

std::vector<BoundingBox> CudaYoloPostproc::process(const std::vector<core::Tensor>& tensors) {
    if (tensors.empty()) return {};

    const auto& output = tensors[0];
    const float* d_data = static_cast<const float*>(output.data());
    
    // Shape: [1, Channels, Anchors]
    // Channels = 4 + NumClasses
    int num_channels = (int)output.shape()[1];
    int num_anchors  = (int)output.shape()[2];
    int num_classes  = num_channels - 4;

    // 1. Reset Counter
    cudaMemsetAsync(d_count, 0, sizeof(int), m_stream);

    // 2. Launch Kernel
    int threads = 256;
    int blocks = (num_anchors + threads - 1) / threads;

    decode_yolov8_kernel<<<blocks, threads, 0, m_stream>>>(
        d_data,
        d_count,
        d_candidates,
        num_classes,
        num_anchors,
        m_config.conf_threshold,
        MAX_GPU_CANDIDATES
    );

    // 3. Download Count (Async)
    cudaMemcpyAsync(h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream); // Wait for count

    int count = *h_count;
    if (count == 0) return {};
    
    // Clamp to max
    if (count > MAX_GPU_CANDIDATES) count = MAX_GPU_CANDIDATES;

    // 4. Download Candidates
    cudaMemcpyAsync(h_candidates, d_candidates, count * sizeof(GpuBox), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream); // Wait for data

    // 5. Final NMS on CPU (Using OpenCV which is highly optimized)
    // Why CPU NMS? 
    // - We filtered 8400 anchors down to maybe 20-50 boxes on GPU.
    // - Sorting and NMSing 50 boxes on CPU is < 0.1ms. 
    // - Writing a robust CUDA NMS is complex and often slower for small N due to sync overhead.
    
    std::vector<cv::Rect> boxes_cv;
    std::vector<float> scores_cv;
    boxes_cv.reserve(count);
    scores_cv.reserve(count);

    for (int i = 0; i < count; ++i) {
        const auto& box = h_candidates[i];
        boxes_cv.emplace_back((int)box.x1, (int)box.y1, (int)(box.x2 - box.x1), (int)(box.y2 - box.y1));
        scores_cv.push_back(box.score);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_cv, scores_cv, m_config.conf_threshold, m_config.nms_threshold, indices);

    // 6. Pack Results
    std::vector<BoundingBox> results;
    results.reserve(indices.size());

    for (int idx : indices) {
        const auto& gpu_box = h_candidates[idx];
        results.push_back({
            gpu_box.x1, gpu_box.y1, gpu_box.x2, gpu_box.y2,
            gpu_box.score,
            (int)gpu_box.class_id
        });
    }

    return results;
}

} // namespace xinfer::postproc