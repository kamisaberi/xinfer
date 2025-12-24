#include <include/postproc/yolo_decoder.h>
#include <cuda_runtime.h>
#include <atomic>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::postproc::yolo {

/**
 * @brief CUDA kernel to decode the raw YOLOv8 output tensor.
 *
 * This kernel processes the [B, C, N] tensor from a YOLO model.
 * B = 1 (batch)
 * C = 4 + num_classes
 * N = num_predictions (e.g., 8400)
 *
 * Each thread processes one of the N potential predictions.
 */
__global__ void decode_yolov8_kernel(
    const float* __restrict__ raw_output,
    float confidence_threshold,
    float* __restrict__ out_boxes,    // [max_detections, 4]
    float* __restrict__ out_scores,   // [max_detections]
    int*   __restrict__ out_classes,  // [max_detections]
    int*   __restrict__ detection_count, // Atomic counter
    int num_predictions,
    int num_classes,
    int max_detections)
{
    // Each thread takes one potential detection
    int pred_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pred_idx >= num_predictions) {
        return;
    }

    // --- 1. Find the class with the highest score ---
    // The class scores start after the 4 box coordinates
    const float* class_scores_ptr = raw_output + 4 * num_predictions + pred_idx;

    float max_score = 0.0f;
    int max_class_id = -1;

    for (int class_id = 0; class_id < num_classes; ++class_id) {
        // Accessing the score for this class: class_id * stride + pred_idx
        float score = class_scores_ptr[class_id * num_predictions];
        if (score > max_score) {
            max_score = score;
            max_class_id = class_id;
        }
    }

    // --- 2. Filter by confidence threshold ---
    if (max_score >= confidence_threshold) {
        // --- 3. If it passes, get the box coordinates and convert them ---
        // The box data is at the beginning of the raw_output tensor
        const float cx = raw_output[0 * num_predictions + pred_idx];
        const float cy = raw_output[1 * num_predictions + pred_idx];
        const float w  = raw_output[2 * num_predictions + pred_idx];
        const float h  = raw_output[3 * num_predictions + pred_idx];

        // Convert from [center_x, center_y, width, height] to [x1, y1, x2, y2]
        float x1 = cx - w / 2.0f;
        float y1 = cy - h / 2.0f;
        float x2 = cx + w / 2.0f;
        float y2 = cy + h / 2.0f;

        // --- 4. Atomically increment the counter and get the write index ---
        // This is a critical step to ensure that multiple threads write to unique
        // locations in the output arrays without overwriting each other.
        int write_idx = atomicAdd(detection_count, 1);

        // --- 5. Write the final data to the output tensors ---
        // Ensure we don't write past the allocated buffer size
        if (write_idx < max_detections) {
            out_boxes[write_idx * 4 + 0] = x1;
            out_boxes[write_idx * 4 + 1] = y1;
            out_boxes[write_idx * 4 + 2] = x2;
            out_boxes[write_idx * 4 + 3] = y2;
            out_scores[write_idx] = max_score;
            out_classes[write_idx] = max_class_id;
        }
    }
}


void decode(const core::Tensor& raw_output,
            float confidence_threshold,
            core::Tensor& out_boxes,
            core::Tensor& out_scores,
            core::Tensor& out_classes)
{
    auto shape = raw_output.shape(); // Expects [1, 4 + num_classes, num_predictions]
    if (shape.size() != 3 || shape[0] != 1) {
        throw std::runtime_error("YOLO decode expects a single-batch tensor of shape [1, C, N]");
    }

    const int num_classes = shape[1] - 4;
    const int num_predictions = shape[2];

    // The maximum number of detections we can store is limited by the size of the output buffers.
    const int max_detections = out_boxes.shape()[0];

    // We need a counter on the GPU to keep track of the number of valid detections.
    // We use a managed memory pointer so we can easily reset it from the CPU.
    int* d_detection_count;
    CHECK_CUDA(cudaMallocManaged(&d_detection_count, sizeof(int)));
    *d_detection_count = 0; // Reset counter

    // --- Launch the Kernel ---
    int threads_per_block = 256;
    int blocks = (num_predictions + threads_per_block - 1) / threads_per_block;

    decode_yolov8_kernel<<<blocks, threads_per_block>>>(
        static_cast<const float*>(raw_output.data()),
        confidence_threshold,
        static_cast<float*>(out_boxes.data()),
        static_cast<float*>(out_scores.data()),
        static_cast<int*>(out_classes.data()),
        d_detection_count,
        num_predictions,
        num_classes,
        max_detections
    );

    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for the kernel to finish

    // After the kernel runs, `*d_detection_count` holds the number of boxes that passed the threshold.
    // A more advanced implementation would use this count to resize the output tensors
    // before passing them to the NMS kernel, to make NMS even faster.
    // For now, we will pass the full-sized buffers.

    CHECK_CUDA(cudaFree(d_detection_count));
}

} // namespace xinfer::postproc::yolo