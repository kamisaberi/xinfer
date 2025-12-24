#pragma once

#include <xinfer/postproc/generative/sampler_interface.h>
#include <cuda_runtime.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CUDA Implementation of Diffusion Sampler (DDIM).
 * 
 * Optimized for Latent Diffusion Models running on TensorRT.
 * 
 * Performance Strategy:
 * 1. Pre-calculates the Schedule (Alphas/Betas) on CPU (negligible size).
 * 2. Launches a Fused CUDA Kernel for the 'step' operation.
 * 3. Keeps Latents in VRAM (Zero D2H copies during the loop).
 */
class CudaDiffusionSampler : public ISamplerPostprocessor {
public:
    CudaDiffusionSampler();
    ~CudaDiffusionSampler() override;

    void init(const SamplerConfig& config) override;

    std::vector<long> set_timesteps(int num_inference_steps) override;

    /**
     * @brief Perform DDIM Step on GPU.
     * 
     * @param model_output Device Pointer (Noise Pred)
     * @param sample       Device Pointer (Current Latent)
     * @param prev_sample  Device Pointer (Next Latent)
     */
    void step(const core::Tensor& model_output, 
              long timestep,
              const core::Tensor& sample, 
              core::Tensor& prev_sample) override;

private:
    SamplerConfig m_config;
    cudaStream_t m_stream = nullptr;

    // --- Schedule Data (CPU) ---
    // We store the full schedule on CPU, but pass only the 
    // specific alpha values for the current timestep to the GPU kernel as scalars.
    std::vector<float> m_alphas_cumprod;
    
    std::vector<long> m_timesteps;
    int m_num_inference_steps = 0;

    void build_schedule();
    float get_alpha_cumprod(long t) const;
};

} // namespace xinfer::postproc