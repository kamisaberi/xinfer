#include <include/postproc/anomaly.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <stdexcept>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::postproc::anomaly {

/**
 * @brief CUDA kernel to compute the pixel-wise squared error.
 *
 * Calculates (original - reconstructed)^2 for each pixel across all channels
 * and outputs a single-channel heat map.
 */
__global__ void reconstruction_error_kernel(
    const float* original,
    const float* reconstructed,
    float* anomaly_map,
    int C, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < W && y < H) {
        float sum_sq_err = 0.0f;
        for (int c = 0; c < C; ++c) {
            int idx = c * H * W + y * W + x;
            float diff = original[idx] - reconstructed[idx];
            sum_sq_err += diff * diff;
        }

        // The anomaly map stores the mean squared error for that pixel
        anomaly_map[y * W + x] = sum_sq_err / C;
    }
}


void calculate_reconstruction_error(
    const core::Tensor& original_image,
    const core::Tensor& reconstructed_image,
    core::Tensor& out_anomaly_map,
    float& out_anomaly_score)
{
    auto in_shape = original_image.shape(); // Expects [B, C, H, W]
    if (in_shape.size() != 4 || in_shape[0] != 1) {
        throw std::runtime_error("Reconstruction error expects a single-batch tensor of shape [1, C, H, W]");
    }
    if (in_shape != reconstructed_image.shape() || in_shape != out_anomaly_map.shape()) {
         throw std::runtime_error("Tensor shape mismatch in calculate_reconstruction_error.");
    }

    const int C = in_shape[1];
    const int H = in_shape[2];
    const int W = in_shape[3];
    const size_t num_pixels = H * W;

    // --- Step 1: Launch the kernel to calculate the pixel-wise anomaly map ---
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    reconstruction_error_kernel<<<grid, block>>>(
        static_cast<const float*>(original_image.data()),
        static_cast<const float*>(reconstructed_image.data()),
        static_cast<float*>(out_anomaly_map.data()),
        C, H, W
    );
    CHECK_CUDA(cudaGetLastError());

    // --- Step 2: Calculate the single overall anomaly score ---
    // The overall score is the mean of the entire anomaly map.
    // We use the powerful Thrust library for an efficient parallel reduction (summation).

    // Create a Thrust device_vector that *wraps* our existing GPU memory
    // without making a copy.
    thrust::device_ptr<float> map_ptr(static_cast<float*>(out_anomaly_map.data()));

    // Perform the parallel sum on the GPU
    float total_sum = thrust::reduce(map_ptr, map_ptr + num_pixels);

    // Calculate the mean and store it in the output variable
    out_anomaly_score = total_sum / num_pixels;
}

} // namespace xinfer::postproc::anomaly