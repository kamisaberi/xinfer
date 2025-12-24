#include "diffusion_sampler.h"
#include <xinfer/core/logging.h>

#include <cmath>
#include <algorithm>
#include <iostream>

namespace xinfer::postproc {

CpuDiffusionSampler::CpuDiffusionSampler() {}
CpuDiffusionSampler::~CpuDiffusionSampler() {}

void CpuDiffusionSampler::init(const SamplerConfig& config) {
    m_config = config;
    build_schedule();
}

void CpuDiffusionSampler::build_schedule() {
    int steps = m_config.num_train_timesteps; // Usually 1000
    m_alphas_cumprod.resize(steps);

    // 1. Generate Betas (Scaled Linear Schedule)
    std::vector<float> betas(steps);
    float start = std::sqrt(m_config.beta_start);
    float end = std::sqrt(m_config.beta_end);
    
    for (int i = 0; i < steps; ++i) {
        // Linspace
        float val = start + (end - start) * ((float)i / (steps - 1));
        betas[i] = val * val;
    }

    // 2. Generate Alphas and Cumprod
    float cumprod = 1.0f;
    for (int i = 0; i < steps; ++i) {
        float alpha = 1.0f - betas[i];
        cumprod *= alpha;
        m_alphas_cumprod[i] = cumprod;
    }
}

std::vector<long> CpuDiffusionSampler::set_timesteps(int num_inference_steps) {
    m_num_inference_steps = num_inference_steps;
    m_timesteps.clear();

    // DDIM Uniform Spacing: 1000 -> 980 -> 960 ...
    int step_ratio = m_config.num_train_timesteps / num_inference_steps;
    
    // Generate steps in descending order (Noise -> Image)
    // DDIM usually adds a bias offset (e.g. +1), simplified here
    for (int i = 0; i < num_inference_steps; ++i) {
        long t = (long)(m_config.num_train_timesteps - 1 - (i * step_ratio));
        m_timesteps.push_back(t);
    }
    
    return m_timesteps;
}

float CpuDiffusionSampler::get_alpha_cumprod(long t) const {
    if (t < 0) return 1.0f; // Initial state (full signal)
    if (t >= m_alphas_cumprod.size()) return 0.0f; // Full noise
    return m_alphas_cumprod[t];
}

void CpuDiffusionSampler::step(const core::Tensor& model_output, 
                               long timestep,
                               const core::Tensor& sample, 
                               core::Tensor& prev_sample) {
    
    // 1. Validation
    if (sample.size() != model_output.size()) {
        XINFER_LOG_ERROR("Sampler: Tensor size mismatch.");
        return;
    }

    size_t count = sample.size();
    if (prev_sample.size() != count) {
        prev_sample.resize(sample.shape(), core::DataType::kFLOAT);
    }

    // 2. Get Schedule Params
    // Calculate previous timestep
    long prev_timestep = timestep - (m_config.num_train_timesteps / m_num_inference_steps);
    
    float alpha_prod_t = get_alpha_cumprod(timestep);
    float alpha_prod_t_prev = get_alpha_cumprod(prev_timestep);
    
    float beta_prod_t = 1.0f - alpha_prod_t;
    float beta_prod_t_prev = 1.0f - alpha_prod_t_prev;

    float sqrt_alpha_prod_t = std::sqrt(alpha_prod_t);
    float sqrt_one_minus_alpha_prod_t = std::sqrt(beta_prod_t);
    
    // 3. DDIM Update Loop
    // Access raw pointers
    const float* eps_ptr = static_cast<const float*>(model_output.data()); // Noise Pred
    const float* sample_ptr = static_cast<const float*>(sample.data());    // Current Latent
    float* prev_ptr = static_cast<float*>(prev_sample.data());             // Next Latent

    // This loop should be auto-vectorized by compiler (AVX/NEON)
    for (size_t i = 0; i < count; ++i) {
        float eps = eps_ptr[i];
        float curr = sample_ptr[i];

        // A. Predict Original Sample (x_0)
        // "What would the image look like if we removed all noise now?"
        float pred_original_sample = (curr - sqrt_one_minus_alpha_prod_t * eps) / sqrt_alpha_prod_t;

        // Clip x_0 (Optional but recommended for stability)
        // pred_original_sample = std::max(-1.0f, std::min(1.0f, pred_original_sample));

        // B. Compute Direction pointing to x_t
        float pred_sample_direction = std::sqrt(beta_prod_t_prev) * eps;

        // C. Compute Previous Sample (x_t-1)
        // Combine predicted original + direction
        float prev = std::sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction;

        prev_ptr[i] = prev;
    }
}

} // namespace xinfer::postproc