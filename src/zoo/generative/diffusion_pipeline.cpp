#include <include/zoo/generative/diffusion_pipeline.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>
#include <cuda_runtime_api.h>
#include <curand.h>

#include <include/core/engine.h>
#include <include/postproc/diffusion_sampler.h>

#define CHECK_CUDA(call) { const cudaError_t e = call; if (e != cudaSuccess) { throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(e))); } }
#define CHECK_CURAND(call) { const curandStatus_t s = call; if (s != CURAND_STATUS_SUCCESS) { throw std::runtime_error("CURAND Error"); } }


namespace xinfer::zoo::generative {

struct DiffusionPipeline::Impl {
    DiffusionPipelineConfig config_;
    std::unique_ptr<core::InferenceEngine> unet_engine_;

    // Scheduler constants on the GPU
    core::Tensor d_betas;
    core::Tensor d_alphas;
    core::Tensor d_alphas_cumprod;

    cudaStream_t stream_;
    curandGenerator_t noise_generator_;

    Impl(const DiffusionPipelineConfig& config) : config_(config) {
        // Pre-calculate scheduler constants on CPU
        std::vector<float> h_betas(config.num_timesteps);
        std::vector<float> h_alphas(config.num_timesteps);
        std::vector<float> h_alphas_cumprod(config.num_timesteps);

        float start = 0.0001f;
        float end = 0.02f;
        for (int i = 0; i < config.num_timesteps; ++i) {
            h_betas[i] = start + (end - start) * i / (config.num_timesteps - 1);
            h_alphas[i] = 1.0f - h_betas[i];
            h_alphas_cumprod[i] = (i > 0) ? h_alphas_cumprod[i-1] * h_alphas[i] : h_alphas[i];
        }

        d_betas = core::Tensor({config.num_timesteps}, core::DataType::kFLOAT);
        d_alphas = core::Tensor({config.num_timesteps}, core::DataType::kFLOAT);
        d_alphas_cumprod = core::Tensor({config.num_timesteps}, core::DataType::kFLOAT);
        d_betas.copy_from_host(h_betas.data());
        d_alphas.copy_from_host(h_alphas.data());
        d_alphas_cumprod.copy_from_host(h_alphas_cumprod.data());
    }
};

DiffusionPipeline::DiffusionPipeline(const DiffusionPipelineConfig& config)
    : pimpl_(new Impl(config))
{
    if (!std::ifstream(pimpl_->config_.unet_engine_path).good()) {
        throw std::runtime_error("Diffusion U-Net engine file not found: " + pimpl_->config_.unet_engine_path);
    }

    pimpl_->unet_engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.unet_engine_path);
    CHECK_CUDA(cudaStreamCreate(&pimpl_->stream_));
    CHECK_CURAND(curandCreateGenerator(&pimpl_->noise_generator_, CURAND_RNG_PSEUDO_DEFAULT));
    CHECK_CURAND(curandSetPseudoRandomGeneratorSeed(pimpl_->noise_generator_, 1234ULL));
}

DiffusionPipeline::~DiffusionPipeline() {
    cudaStreamDestroy(pimpl_->stream_);
    curandDestroyGenerator(pimpl_->noise_generator_);
}

DiffusionPipeline::DiffusionPipeline(DiffusionPipeline&&) noexcept = default;
DiffusionPipeline& DiffusionPipeline::operator=(DiffusionPipeline&&) noexcept = default;

core::Tensor DiffusionPipeline::generate(int batch_size) {
    if (!pimpl_) throw std::runtime_error("DiffusionPipeline is in a moved-from state.");

    auto unet_input_shape = pimpl_->unet_engine_->get_input_shape(0);
    unet_input_shape[0] = batch_size;

    core::Tensor img_tensor(unet_input_shape, core::DataType::kFLOAT);
    CHECK_CURAND(curandGenerateNormal((float*)img_tensor.data(), img_tensor.num_elements(), 0.0f, 1.0f));

    core::Tensor time_tensor({batch_size}, core::DataType::kFLOAT);
    core::Tensor random_noise_tensor(unet_input_shape, core::DataType::kFLOAT);

    for (int i = pimpl_->config_.num_timesteps - 1; i >= 0; --i) {
        std::vector<float> h_time(batch_size, (float)i);
        time_tensor.copy_from_host(h_time.data());

        auto output_tensors = pimpl_->unet_engine_->infer({img_tensor, time_tensor});
        const core::Tensor& predicted_noise = output_tensors[0];

        if (i > 0) {
            CHECK_CURAND(curandGenerateNormal((float*)random_noise_tensor.data(), random_noise_tensor.num_elements(), 0.0f, 1.0f));
        } else {
            CHECK_CUDA(cudaMemsetAsync(random_noise_tensor.data(), 0, random_noise_tensor.size_bytes(), pimpl_->stream_));
        }

        postproc::diffusion::sampling_step(
            img_tensor,
            predicted_noise,
            random_noise_tensor,
            pimpl_->d_alphas,
            pimpl_->d_alphas_cumprod,
            pimpl_->d_betas,
            i,
            pimpl_->stream_
        );
    }
    CHECK_CUDA(cudaStreamSynchronize(pimpl_->stream_));

    return img_tensor;
}

} // namespace xinfer::zoo::generative