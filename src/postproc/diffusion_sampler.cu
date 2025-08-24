#include <include/postproc/diffusion_sampler.h>
#include <cuda_runtime.h>
#include <stdexcept>

#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(error))); \
    } \
}

namespace xinfer::postproc::diffusion {

/**
 * @brief Fused CUDA kernel for a single DDPM sampling step.
 */
__global__ void sampling_kernel(
    float* __restrict__ img,
    const float* __restrict__ predicted_noise,
    const float* __restrict__ random_noise,
    float alpha_t,
    float alpha_t_cumprod,
    float beta_t,
    size_t num_elements)
{
    // Using a standard grid-stride loop to process all elements
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_elements;
         i += blockDim.x * gridDim.x)
    {
        // Pre-calculate coefficients on the CPU and pass them in for efficiency
        float inv_sqrt_alpha_t = rsqrtf(alpha_t);
        float noise_coeff = (1.0f - alpha_t) / sqrtf(1.0f - alpha_t_cumprod);
        float sigma = sqrtf(beta_t);

        // Perform the entire sampling step equation in one go
        img[i] = inv_sqrt_alpha_t * (img[i] - noise_coeff * predicted_noise[i]) + sigma * random_noise[i];
    }
}

void sampling_step(
    core::Tensor& img,
    const core::Tensor& predicted_noise,
    const core::Tensor& random_noise,
    const core::Tensor& alphas,
    const core::Tensor& alphas_cumprod,
    const core::Tensor& betas,
    int timestep,
    cudaStream_t stream)
{
    if (img.shape() != predicted_noise.shape() || img.shape() != random_noise.shape()) {
        throw std::invalid_argument("All tensors in sampling_step must have the same shape.");
    }

    // --- Get the specific scheduler constants for this timestep ---
    // This involves a very small GPU->CPU copy for three float values.
    // This is much faster than passing the entire scheduler tensor to the kernel.
    float h_alpha_t, h_alpha_t_cumprod, h_beta_t;

    CHECK_CUDA(cudaMemcpyAsync(&h_alpha_t,
                               static_cast<const float*>(alphas.data()) + timestep,
                               sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&h_alpha_t_cumprod,
                               static_cast<const float*>(alphas_cumprod.data()) + timestep,
                               sizeof(float), cudaMemcpyDeviceToHost, stream));
    CHECK_CUDA(cudaMemcpyAsync(&h_beta_t,
                               static_cast<const float*>(betas.data()) + timestep,
                               sizeof(float), cudaMemcpyDeviceToHost, stream));

    // We must wait for these tiny copies to finish before launching the kernel
    CHECK_CUDA(cudaStreamSynchronize(stream));

    // --- Launch the fused kernel ---
    size_t num_elements = img.num_elements();
    int threads_per_block = 256;
    int blocks = (num_elements + threads_per_block - 1) / threads_per_block;

    sampling_kernel<<<blocks, threads_per_block, 0, stream>>>(
        static_cast<float*>(img.data()),
        static_cast<const float*>(predicted_noise.data()),
        static_cast<const float*>(random_noise.data()),
        h_alpha_t,
        h_alpha_t_cumprod,
        h_beta_t,
        num_elements
    );
    CHECK_CUDA(cudaGetLastError());
}

} // namespace xinfer::postproc::diffusion