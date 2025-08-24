#pragma once

#include <include/core/tensor.h>
#include <cuda_runtime_api.h> // For cudaStream_t

namespace xinfer::postproc::diffusion {

    /**
     * @brief Performs one step of the DDPM denoising/sampling process.
     *
     * This function executes a single, fused CUDA kernel to apply the reverse
     * diffusion equation for one timestep. It takes the current noisy image and
     * the model's predicted noise, and outputs the slightly less noisy image for
     * the next step (t-1).
     *
     * The formula implemented is:
     * img_{t-1} = 1/sqrt(alpha_t) * (img_t - (1-alpha_t)/sqrt(1-alpha_cumprod_t) * pred_noise) + sqrt(beta_t) * random_noise
     *
     * @param img The current noisy image tensor (on GPU). This tensor is modified in-place.
     * @param predicted_noise The noise tensor predicted by the U-Net model for the current step (on GPU).
     * @param random_noise A tensor of random Gaussian noise for the stochastic part of the step (on GPU).
     * @param alphas A GPU tensor containing all alpha scheduler constants.
     * @param alphas_cumprod A GPU tensor containing all cumulative alpha products.
     * @param betas A GPU tensor containing all beta scheduler constants.
     * @param timestep The current timestep index (t).
     * @param stream The CUDA stream to execute the kernel on.
     */
    void sampling_step(
        core::Tensor& img,
        const core::Tensor& predicted_noise,
        const core::Tensor& random_noise,
        const core::Tensor& alphas,
        const core::Tensor& alphas_cumprod,
        const core::Tensor& betas,
        int timestep,
        cudaStream_t stream
    );

} // namespace xinfer::postproc::diffusion

#endif // XINFER_POSTPROC_DIFFUSION_SAMPLER_H