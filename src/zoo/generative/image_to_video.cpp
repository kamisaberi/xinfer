#include <xinfer/zoo/generative/image_to_video.h>
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

struct ImageToVideo::Impl {
    VideoGenConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> image_encoder_;
    std::unique_ptr<backends::IBackend> unet_;
    std::unique_ptr<backends::IBackend> vae_decoder_;

    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISamplerPostprocessor> sampler_;

    // --- Tensors ---
    // VAE Encode
    core::Tensor img_enc_input;

    // UNet
    core::Tensor unet_latents; // [Batch, Time, C, H, W]
    core::Tensor unet_timestep;

    // PRNG
    std::mt19937 rng_;

    Impl(const VideoGenConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Engines
        image_encoder_ = backends::BackendFactory::create(config_.target);
        unet_ = backends::BackendFactory::create(config_.target);
        vae_decoder_ = backends::BackendFactory::create(config_.target);

        if (!image_encoder_->load_model(config_.image_encoder_path)) throw std::runtime_error("Failed to load VAE Encoder.");
        if (!unet_->load_model(config_.video_unet_path)) throw std::runtime_error("Failed to load Video UNet.");
        if (!vae_decoder_->load_model(config_.vae_decoder_path)) throw std::runtime_error("Failed to load VAE Decoder.");

        // 2. Setup Preprocessor (for initial image)
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.width;
        pre_cfg.target_height = config_.height;
        // Normalize to [-1, 1] for VAE
        pre_cfg.norm_params.scale_factor = 1.0f / 127.5f;
        pre_cfg.norm_params.mean = {127.5f, 127.5f, 127.5f};
        preproc_->init(pre_cfg);

        // 3. Setup Sampler
        sampler_ = postproc::create_sampler(config_.target);
        postproc::SamplerConfig samp_cfg;
        sampler_->init(samp_cfg);

        // 4. Setup PRNG
        if (config_.seed != -1) rng_.seed(config_.seed);
        else { std::random_device rd; rng_.seed(rd()); }
    }

    cv::Mat tensor_to_image(const float* data, int h, int w) {
        // Helper to decode a single frame from the batch VAE output
        cv::Mat result(h, w, CV_8UC3);
        int plane_size = h * w;

        for (int i = 0; i < plane_size; ++i) {
            // Denormalize [-1, 1] -> [0, 255]
            float r = (data[0 * plane_size + i] * 0.5f + 0.5f) * 255.0f;
            float g = (data[1 * plane_size + i] * 0.5f + 0.5f) * 255.0f;
            float b = (data[2 * plane_size + i] * 0.5f + 0.5f) * 255.0f;

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

ImageToVideo::ImageToVideo(const VideoGenConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ImageToVideo::~ImageToVideo() = default;
ImageToVideo::ImageToVideo(ImageToVideo&&) noexcept = default;
ImageToVideo& ImageToVideo::operator=(ImageToVideo&&) noexcept = default;

std::vector<cv::Mat> ImageToVideo::generate(const cv::Mat& init_image) {
    if (!pimpl_) throw std::runtime_error("ImageToVideo is null.");

    // --- 1. VAE Encode Initial Image ---
    // Preprocess
    preproc::ImageFrame frame{init_image.data, init_image.cols, init_image.rows, preproc::ImageFormat::BGR};
    pimpl_->preproc_->process(frame, pimpl_->img_enc_input);

    // Encode
    core::Tensor initial_latent;
    pimpl_->image_encoder_->predict({pimpl_->img_enc_input}, {initial_latent});
    // Scale by VAE factor
    // tensor_mul_scalar(initial_latent, 0.18215f);

    // --- 2. Prepare for Denoising Loop ---
    int T = pimpl_->config_.num_frames;
    int C = initial_latent.shape()[1];
    int H = initial_latent.shape()[2];
    int W = initial_latent.shape()[3];

    // Create Noise Tensor
    core::Tensor noise({1, (int64_t)T, (int64_t)C, (int64_t)H, (int64_t)W}, core::DataType::kFLOAT);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    float* n_ptr = static_cast<float*>(noise.data());
    for(size_t i=0; i<noise.size(); ++i) n_ptr[i] = dist(pimpl_->rng_);

    // Set initial state for sampler (Noise)
    core::Tensor latents = noise.clone();

    // Set Timesteps
    auto timesteps = pimpl_->sampler_->set_timesteps(pimpl_->config_.num_inference_steps);

    // --- 3. Denoising Loop ---
    for (long t : timesteps) {
        // A. Add initial image latent to current noise state (conditioning)
        // This is a key part of I2V models
        // latents = latents + initial_latent_broadcasted

        // B. UNet Inference
        // Input: [Latents, Timestep, Image_Embedding (Initial Latent)]
        core::Tensor t_tensor({1}, core::DataType::kINT32);
        ((int*)t_tensor.data())[0] = (int)t;

        core::Tensor noise_pred;
        pimpl_->unet_->predict({latents, t_tensor, initial_latent}, {noise_pred});

        // C. Sampling Step
        core::Tensor next_latents;
        pimpl_->sampler_->step(noise_pred, t, latents, next_latents);
        latents = next_latents;
    }

    // --- 4. VAE Decode Batch ---
    // Decode all frames at once for speed
    // Input is [T, C, H, W] -> Output is [T, 3, Orig_H, Orig_W]

    // Reshape latents for batch decoding
    // [1, T, C, H, W] -> [T, C, H, W]
    latents.reshape({(int64_t)T, (int64_t)C, (int64_t)H, (int64_t)W}, core::DataType::kFLOAT);

    core::Tensor decoded_frames;
    pimpl_->vae_decoder_->predict({latents}, {decoded_frames});

    // --- 5. Convert to OpenCV Frames ---
    std::vector<cv::Mat> video_frames;

    auto shape = decoded_frames.shape();
    int out_h = shape[2];
    int out_w = shape[3];
    size_t frame_size = 3 * out_h * out_w;
    const float* data = static_cast<const float*>(decoded_frames.data());

    for (int i = 0; i < T; ++i) {
        // Point to start of frame i
        const float* frame_data = data + (i * frame_size);
        video_frames.push_back(pimpl_->tensor_to_image(frame_data, out_h, out_w));
    }

    return video_frames;
}

} // namespace xinfer::zoo::generative