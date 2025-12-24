#include "diffusion_sampler.cuh"
#include <xinfer/core/logging.h>
#include <cmath>
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernel: Fused DDIM Step
// =================================================================================

/**
 * @brief Performs one step of denoising.
 * 
 * Calculates x_prev based on x_t and noise_pred.
 * Math is fused to minimize global memory reads/writes.
 */
__global__ void ddim_step_kernel(const float* __restrict__ model_output, // epsilon
                                 const float* __restrict__ sample,       // x_t
                                 float* __restrict__ prev_sample,        // x_t-1
                                 float alpha_prod_t,
                                 float alpha_prod_t_prev,
                                 int num_elements) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    // 1. Load Data
    float eps = model_output[idx];
    float curr = sample[idx];

    // 2. Pre-calculate Derived Constants (Could be passed in, but cheap to compute)
    float beta_prod_t = 1.0f - alpha_prod_t;
    // float beta_prod_t_prev = 1.0f - alpha_prod_t_prev; // Not strictly used in simple DDIM form below

    float sqrt_alpha_prod_t = sqrtf(alpha_prod_t);
    float sqrt_one_minus_alpha_prod_t = sqrtf(beta_prod_t);

    // 3. Predict Original Sample (x_0)
    // "remove the noise from current sample based on model prediction"
    float pred_original_sample = (curr - sqrt_one_minus_alpha_prod_t * eps) / sqrt_alpha_prod_t;

    // Optional: Clip x_0 to [-1, 1] for numerical stability
    // pred_original_sample = fmaxf(-1.0f, fminf(1.0f, pred_original_sample));

    // 4. Compute Direction pointing to x_t
    // For DDIM (deterministic), noise is 0, so direction is purely defined by schedule
    float pred_sample_direction = sqrtf(1.0f - alpha_prod_t_prev) * eps;

    // 5. Compute Previous Sample (x_t-1)
    float prev = sqrtf(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction;

    // 6. Write Result
    prev_sample[idx] = prev;
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaDiffusionSampler::CudaDiffusionSampler() {
    cudaStreamCreate(&m_stream);
}

CudaDiffusionSampler::~CudaDiffusionSampler() {
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaDiffusionSampler::init(const SamplerConfig& config) {
    m_config = config;
    build_schedule();
}

void CudaDiffusionSampler::build_schedule() {
    // Exact same logic as CPU version, kept on Host
    int steps = m_config.num_train_timesteps; 
    m_alphas_cumprod.resize(steps);

    std::vector<float> betas(steps);
    float start = std::sqrt(m_config.beta_start);
    float end = std::sqrt(m_config.beta_end);
    
    for (int i = 0; i < steps; ++i) {
        float val = start + (end - start) * ((float)i / (steps - 1));
        betas[i] = val * val;
    }

    float cumprod = 1.0f;
    for (int i = 0; i < steps; ++i) {
        float alpha = 1.0f - betas[i];
        cumprod *= alpha;
        m_alphas_cumprod[i] = cumprod;
    }
}

std::vector<long> CudaDiffusionSampler::set_timesteps(int num_inference_steps) {
    m_num_inference_steps = num_inference_steps;
    m_timesteps.clear();
    int step_ratio = m_config.num_train_timesteps / num_inference_steps;
    
    for (int i = 0; i < num_inference_steps; ++i) {
        long t = (long)(m_config.num_train_timesteps - 1 - (i * step_ratio));
        m_timesteps.push_back(t);
    }
    return m_timesteps;
}

float CudaDiffusionSampler::get_alpha_cumprod(long t) const {
    if (t < 0) return 1.0f;
    if (t >= m_alphas_cumprod.size()) return 0.0f;
    return m_alphas_cumprod[t];
}

void CudaDiffusionSampler::step(const core::Tensor& model_output, 
                                long timestep,
                                const core::Tensor& sample, 
                                core::Tensor& prev_sample) {
    // 1. Validation
    if (sample.size() != model_output.size()) {
        XINFER_LOG_ERROR("CudaSampler: Tensor size mismatch.");
        return;
    }

    size_t count = sample.size();
    if (prev_sample.size() != count) {
        prev_sample.resize(sample.shape(), core::DataType::kFLOAT);
    }

    // Ensure memory is on GPU
    if (sample.memory_type() != core::MemoryType::CudaDevice) {
        XINFER_LOG_WARN_ONCE("CudaSampler: Inputs are not on GPU! Performance will be poor due to implicit copies.");
        // In a real framework, we'd assert here or handle copy
    }

    // 2. Get Schedule Scalars (Host Side)
    long prev_timestep = timestep - (m_config.num_train_timesteps / m_num_inference_steps);
    float alpha_t = get_alpha_cumprod(timestep);
    float alpha_prev = get_alpha_cumprod(prev_timestep);

    // 3. Get Device Pointers
    const float* d_model_out = static_cast<const float*>(model_output.data());
    const float* d_sample    = static_cast<const float*>(sample.data());
    float* d_prev            = static_cast<float*>(prev_sample.data());

    // 4. Launch Kernel
    int threads = 256;
    int blocks = (count + threads - 1) / threads;

    ddim_step_kernel<<<blocks, threads, 0, m_stream>>>(
        d_model_out,
        d_sample,
        d_prev,
        alpha_t,
        alpha_prev,
        (int)count
    );

    // Optional: Synchronize if debugging, otherwise async
    // cudaStreamSynchronize(m_stream);
}

} // namespace xinfer::postproc