#pragma once

#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::postproc {

enum class SchedulerType {
    DDIM = 0,
    EULER_DISCRETE = 1,
    PNDM = 2
};

struct SamplerConfig {
    SchedulerType type = SchedulerType::DDIM;
    
    // Diffusion Parameters (Standard Stable Diffusion 1.5 defaults)
    int num_train_timesteps = 1000;
    float beta_start = 0.00085f;
    float beta_end = 0.012f;
    
    // "scaled_linear" is standard for SD
    bool trained_betas = true; 
};

class ISamplerPostprocessor {
public:
    virtual ~ISamplerPostprocessor() = default;

    virtual void init(const SamplerConfig& config) = 0;

    /**
     * @brief Configure the inference schedule.
     * Must be called before the sampling loop starts.
     * 
     * @param num_inference_steps e.g., 20 or 50.
     * @return Vector of timesteps to loop over (e.g., 981, 961, 941...).
     */
    virtual std::vector<long> set_timesteps(int num_inference_steps) = 0;

    /**
     * @brief Perform one denoising step (x_t -> x_t-1).
     * 
     * @param model_output The predicted noise (epsilon) from UNet.
     *                     Assumes CFG (Guidance) has already been applied.
     * @param timestep     Current timestep t.
     * @param sample       Current latent x_t.
     * @param prev_sample  Output latent x_t-1.
     */
    virtual void step(const core::Tensor& model_output, 
                      long timestep,
                      const core::Tensor& sample, 
                      core::Tensor& prev_sample) = 0;
};

} // namespace xinfer::postproc