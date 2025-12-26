#include <xinfer/zoo/generative/inpainter.h>
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

struct Inpainter::Impl {
    InpainterConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> text_encoder_;
    std::unique_ptr<backends::IBackend> unet_;
    std::unique_ptr<backends::IBackend> vae_encoder_;
    std::unique_ptr<backends::IBackend> vae_decoder_;

    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<preproc::IImagePreprocessor> image_preproc_;
    std::unique_ptr<postproc::ISamplerPostprocessor> sampler_;

    // --- PRNG ---
    std::mt19937 rng_;

    Impl(const InpainterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Load all 4 engines (Text Enc, UNet, VAE Enc, VAE Dec)
        // ... (Code is identical to DiffusionPipeline, omitted for brevity) ...

        // Setup Sampler
        sampler_ = postproc::create_sampler(config_.target);
        postproc::SamplerConfig samp_cfg;
        sampler_->init(samp_cfg);

        // Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::GPT_BPE, config_.target);
        // ... (init tokenizer) ...

        // Setup Image Preprocessor
        image_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.width;
        pre_cfg.target_height = config_.height;
        pre_cfg.norm_params.scale_factor = 1.0f/127.5f;
        pre_cfg.norm_params.mean = {127.5, 127.5, 127.5};
        image_preproc_->init(pre_cfg);

        // Setup PRNG
        if (config_.seed != -1) rng_.seed(config_.seed);
        else { std::random_device rd; rng_.seed(rd()); }
    }

    void encode_prompt(const std::string& prompt, core::Tensor& embeddings) {
        // ... (Code is identical to DiffusionPipeline) ...
    }

    cv::Mat tensor_to_image(const core::Tensor& tensor) {
        // ... (Code is identical to DiffusionPipeline) ...
    }
};

// =================================================================================
// Public API
// =================================================================================

Inpainter::Inpainter(const InpainterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Inpainter::~Inpainter() = default;
Inpainter::Inpainter(Inpainter&&) noexcept = default;
Inpainter& Inpainter::operator=(Inpainter&&) noexcept = default;

cv::Mat Inpainter::inpaint(const cv::Mat& image,
                           const cv::Mat& mask,
                           const std::string& prompt,
                           const std::string& negative_prompt) {
    if (!pimpl_) throw std::runtime_error("Inpainter is null.");

    int latent_h = pimpl_->config_.height / 8;
    int latent_w = pimpl_->config_.width / 8;

    // --- 1. Encode Image and Mask ---
    core::Tensor img_tensor, original_latent, masked_latent;

    // Preprocess image
    preproc::ImageFrame frame{image.data, image.cols, image.rows, preproc::ImageFormat::BGR};
    pimpl_->image_preproc_->process(frame, img_tensor);

    // VAE Encode
    pimpl_->vae_encoder_->predict({img_tensor}, {original_latent});
    // Scale latent
    // original_latent *= 0.18215f;

    // Prepare Mask
    // Resize mask to latent dimensions
    cv::Mat mask_gray, mask_latent_cv;
    if (mask.channels() == 3) cv::cvtColor(mask, mask_gray, cv::COLOR_BGR2GRAY);
    else mask_gray = mask;
    cv::resize(mask_gray, mask_latent_cv, cv::Size(latent_w, latent_h), 0, 0, cv::INTER_NEAREST);
    cv::threshold(mask_latent_cv, mask_latent_cv, 127, 255, cv::THRESH_BINARY);

    core::Tensor mask_latent({1, 1, (int64_t)latent_h, (int64_t)latent_w}, core::DataType::kFLOAT);
    // Convert cv::Mat to tensor and normalize 0-1
    // ... (logic omitted for brevity) ...

    // Create masked image latent
    // masked_latent = original_latent * (1.0 - mask_latent)

    // --- 2. Encode Prompts ---
    core::Tensor pos_embeds, neg_embeds, combined_embeds;
    pimpl_->encode_prompt(prompt, pos_embeds);
    pimpl_->encode_prompt(negative_prompt.empty() ? "" : negative_prompt, neg_embeds);
    // Concatenate pos_embeds and neg_embeds into combined_embeds

    // --- 3. Denoising Loop ---
    // Prepare initial noise
    core::Tensor latents({1, 4, (int64_t)latent_h, (int64_t)latent_w}, core::DataType::kFLOAT);
    // ... (fill with gaussian noise) ...

    auto timesteps = pimpl_->sampler_->set_timesteps(pimpl_->config_.num_inference_steps);

    for (long t : timesteps) {
        // Create 9-channel UNet input
        core::Tensor unet_input; // Shape [2, 9, H, W]
        // Channel order: [noisy_latent(4), mask(1), masked_original_latent(4)]
        // Concatenate tensors
        // ... (logic omitted for brevity) ...

        core::Tensor t_tensor, noise_pred_batch;
        pimpl_->unet_->predict({unet_input, t_tensor, combined_embeds}, {noise_pred_batch});

        // CFG Guidance
        // ... (logic identical to DiffusionPipeline) ...

        // Sampler Step
        core::Tensor next_latents;
        pimpl_->sampler_->step(/*guided_noise*/, t, latents, next_latents);
        latents = next_latents;
    }

    // --- 4. VAE Decode ---
    core::Tensor final_image_tensor;
    pimpl_->vae_decoder_->predict({latents}, {final_image_tensor});

    // --- 5. Finalize ---
    // The model only generates the masked region. We must blend it back.
    cv::Mat generated_img = pimpl_->tensor_to_image(final_image_tensor);
    cv::Mat result_img = image.clone();

    // Resize generated to original size and mask to original size
    cv::resize(generated_img, generated_img, image.size());
    cv::Mat final_mask;
    cv::resize(mask_gray, final_mask, image.size(), 0, 0, cv::INTER_NEAREST);

    // Copy generated region into original image using the mask
    generated_img.copyTo(result_img, final_mask);

    return result_img;
}

} // namespace xinfer::zoo::generative