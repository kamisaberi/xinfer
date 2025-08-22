#include <include/postproc/segmentation.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) { /* ... */ }

// --- The ArgMax CUDA Kernel ---
// This kernel takes a CHW tensor and finds the index of the maximum value
// along the C dimension for each (H, W) pixel.
__global__ void argmax_kernel(const float* logits, int* output_mask, int C, int H, int W) {
    // Each thread handles one pixel (H, W)
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        float max_val = -1e20f;
        int max_idx = -1;

        // Iterate through all channels for this pixel
        for (int c = 0; c < C; ++c) {
            // Logits are in CHW format
            float val = logits[c * H * W + y * W + x];
            if (val > max_val) {
                max_val = val;
                max_idx = c;
            }
        }

        // Write the index of the max channel to the output mask
        output_mask[y * W + x] = max_idx;
    }
}


namespace xinfer::postproc {

void argmax(const core::Tensor& logits, core::Tensor& output_mask) {
    auto logit_shape = logits.shape(); // Expects [B, C, H, W]
    if (logit_shape.size() != 4 || logit_shape[0] != 1) {
        throw std::runtime_error("ArgMax expects a single-batch tensor of shape [1, C, H, W]");
    }

    const int C = logit_shape[1];
    const int H = logit_shape[2];
    const int W = logit_shape[3];

    // Check that the output tensor is correctly sized
    auto mask_shape = output_mask.shape();
    if (mask_shape.size() != 3 || mask_shape[0] != 1 || mask_shape[1] != H || mask_shape[2] != W) {
        throw std::runtime_error("Output mask for ArgMax has incorrect shape.");
    }

    // --- Launch the kernel ---
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    argmax_kernel<<<grid, block>>>(
        static_cast<const float*>(logits.data()),
        static_cast<int*>(output_mask.data()),
        C, H, W
    );
}

cv::Mat argmax_to_mat(const core::Tensor& logits) {
    auto logit_shape = logits.shape();
    if (logit_shape.size() != 4 || logit_shape[0] != 1) {
        throw std::runtime_error("argmax_to_mat expects a single-batch tensor of shape [1, C, H, W]");
    }

    const int H = logit_shape[2];
    const int W = logit_shape[3];

    // Create a temporary GPU tensor to hold the integer mask
    core::Tensor d_mask({1, H, W}, core::DataType::kINT32);

    // Run the GPU-side argmax
    argmax(logits, d_mask);

    // Create a CPU-side cv::Mat to hold the final result
    cv::Mat h_mask(H, W, CV_32S);

    // Copy the result from the GPU to the CPU
    d_mask.copy_to_host(h_mask.data);

    return h_mask;
}

// Implementations for apply_color_map and process_instance_masks would follow a similar pattern.

} // namespace xinfer::postproc