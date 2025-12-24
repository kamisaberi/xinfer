#pragma once

#include <xinfer/postproc/generative/sampler_interface.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CPU Implementation of Diffusion Sampler (DDIM).
 * 
 * Optimized for CPU execution using vectorized loops.
 * While UNet inference usually happens on NPU/GPU, the sampling math 
 * (which is just element-wise addition/multiplication) is often done on CPU 
 * to save development time on custom NPU kernels.
 */
class CpuDiffusionSampler : public ISamplerPostprocessor {
public:
    CpuDiffusionSampler();
    ~CpuDiffusionSampler() override;

    void init(const SamplerConfig& config) override;

    std::vector<long> set_timesteps(int num_inference_steps) override;

    void step(const core::Tensor& model_output, 
              long timestep,
              const core::Tensor& sample, 
              core::Tensor& prev_sample) override;

private:
    SamplerConfig m_config;

    // --- Precomputed Schedule ---
    // Alpha Cumulative Products (bar_alpha)
    std::vector<float> m_alphas_cumprod;
    
    // The timesteps for the current inference run
    std::vector<long> m_timesteps;
    int m_num_inference_steps = 0;

    // Helper: Calculate betas and alphas on init
    void build_schedule();
    
    // Helper: Get alpha_cumprod for a specific timestep
    float get_alpha_cumprod(long t) const;
};

} // namespace xinfer::postproc