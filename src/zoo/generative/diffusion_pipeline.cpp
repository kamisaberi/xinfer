#include <xinfer/zoo/generative/diffusion_pipeline.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/generative/sampler_interface.h>

#include <iostream>
#include <random>
#include <vector>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DiffusionPipeline::Impl {
    DiffusionConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> text_encoder_;
    std::unique_ptr<backends::IBackend> unet_;
    std::unique_ptr<backends::IBackend> vae_decoder_;

    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::ISamplerPostprocessor> sampler_;

    // --- Tensors ---
    // Text Encoder
    core::Tensor text_input_ids, text_attn_mask, text_embeddings;

    // UNet
    core::Tensor unet_latents, unet_timestep, unet_noise_pred;

    // VAE
    core::Tensor vae_output_image;

    // PRNG
    std::mt19937 rng_;

    Impl(const DiffusionConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Engines
        text_encoder_ = backends::BackendFactory::create(config_.target);
        unet_ = backends::BackendFactory::create(config_.target);
        vae_decoder_ = backends::BackendFactory::create(config_.target);

        if (!text_encoder_->load_model(config_.text_encoder_path)) throw std::runtime_error("Failed to load Text Encoder.");
        if (!unet_->load_model(config_.unet_path)) throw std::runtime_error("Failed to load UNet.");
        if (!vae_decoder_->load_model(config_.vae_decoder_path)) throw std::runtime_error("Failed to load VAE.");

        // 2. Setup Tokenizer (BPE for CLIP)
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::GPT_BPE, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.merges_path = config_.merges_path;
        tok_cfg.max_length = 77; // CLIP standard
        tokenizer_->init(tok_cfg);

        // 3. Setup Sampler
        sampler_ = postproc::create_sampler(config_.target);
        postproc::SamplerConfig samp_cfg;
        sampler_->init(samp_cfg);

        // 4. Setup PRNG
        if (config_.seed != -1) rng_.seed(config_.seed);
        else { std::random_device rd; rng_.seed(rd()); }
    }

    // --- Helper: Generate Text Embeddings ---
    void encode_prompt(const std::string& prompt, core::Tensor& embeddings) {
        tokenizer_->process(prompt, text_input_ids, text_attn_mask);
        text_encoder_->predict({text_input_ids}, {embeddings});
    }

    // --- Helper: Post-process VAE output ---
    cv::Mat tensor_to_image(const core::Tensor& tensor) {
        // VAE output is [1, 3, H, W] in [-1, 1] range.
        // Convert to OpenCV Mat [H, W, 3] in [0, 255] uint8 range.

        auto shape = tensor.shape();
        int c = shape[1], h = shape[2], w = shape[3];
        const float* data = static_cast<const float*>(tensor.data());

        cv::Mat result(h, w, CV_8UC3);
        int plane_size = h * w;

        for (int i = 0; i < plane_size; ++i) {
            float r = (data[0 * plane_size + i] * 0.5f + 0.5f) * 255.0f;
            float g = (data[1 * plane_size + i] * 0.5f + 0.5f) * 255.0f;
            float b = (data[2 * plane_size + i] * 0.5f + 0.5f);

            result.at<cv::Vec3b>(i) = cv::Vec3b(
                (uint8_t)std::clamp(b, 0.0f, 255.0f),
                (uint8_t)std::clamp(g, 0.0f, 255.0f),
                (uint8_t)std::clamp(r, 0.0f, 255.0f)
            );
        }
        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DiffusionPipeline::DiffusionPipeline(const DiffusionConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DiffusionPipeline::~DiffusionPipeline() = default;
DiffusionPipeline::DiffusionPipeline(DiffusionPipeline&&) noexcept = default;
DiffusionPipeline& DiffusionPipeline::operator=(DiffusionPipeline&&) noexcept = default;

cv::Mat DiffusionPipeline::generate(const std::string& prompt,
                                    const std::string& negative_prompt,
                                    ProgressCallback progress_callback) {
    if (!pimpl_) throw std::runtime_error("DiffusionPipeline is null.");

    // 1. Encode Prompts
    core::Tensor pos_embeds, neg_embeds;
    pimpl_->encode_prompt(prompt, pos_embeds);
    pimpl_->encode_prompt(negative_prompt.empty() ? "" : negative_prompt, neg_embeds);

    // Concatenate for CFG
    // [1, 77, 768] + [1, 77, 768] -> [2, 77, 768]
    // (This requires Tensor::concat or manual memcpy)
    // For simplicity, we assume the UNet can take two separate embedding tensors
    // or we handle concatenation here.

    // 2. Prepare Latents
    int latent_h = pimpl_->config_.height / 8;
    int latent_w = pimpl_->config_.width / 8;
    core::Tensor latents({1, 4, (int64_t)latent_h, (int64_t)latent_w}, core::DataType::kFLOAT);

    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* l_ptr = static_cast<float*>(latents.data());
    for(size_t i=0; i<latents.size(); ++i) l_ptr[i] = dist(pimpl_->rng_);

    // 3. Denoising Loop
    auto timesteps = pimpl_->sampler_->set_timesteps(pimpl_->config_.num_inference_steps);
    int step_count = 0;

    for (long t : timesteps) {
        // --- Classifier-Free Guidance ---
        // Duplicate latents for batch of 2
        core::Tensor latent_model_input; // [2, 4, H, W]
        // (Concatenate latents with itself - logic omitted)

        // UNet Inference (Batch of 2)
        core::Tensor t_tensor({2}, core::DataType::kINT32);
        ((int*)t_tensor.data())[0] = (int)t;
        ((int*)t_tensor.data())[1] = (int)t;

        // Concatenated Embeds [2, 77, 768]
        core::Tensor combined_embeds;

        pimpl_->unet_->predict({latent_model_input, t_tensor, combined_embeds}, {pimpl_->unet_noise_pred});

        // Split predictions
        // noise_pred_uncond = unet_noise_pred[0]
        // noise_pred_text   = unet_noise_pred[1]
        // (Tensor splitting logic omitted)

        // Perform Guidance
        // guided_noise = uncond + guidance_scale * (text - uncond)

        // --- Step ---
        core::Tensor next_latents;
        pimpl_->sampler_->step(/*guided_noise*/, t, latents, next_latents);
        latents = next_latents; // Update

        if (progress_callback) {
            progress_callback(++step_count, timesteps.size(), latents);
        }
    }

    // 4. VAE Decode
    // latents = 1 / 0.18215 * latents (scaling factor)
    pimpl_->vae_decoder_->predict({latents}, {pimpl_->vae_output_image});

    // 5. Post-process to Image
    return pimpl_->tensor_to_image(pimpl_->vae_output_image);
}

} // namespace xinfer::zoo::generative