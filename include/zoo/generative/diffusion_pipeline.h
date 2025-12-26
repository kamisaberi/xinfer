#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    /**
     * @brief High-level configuration for the Stable Diffusion pipeline.
     */
    struct DiffusionConfig {
        // Hardware Target (GPU with >8GB VRAM is highly recommended)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        std::string text_encoder_path;
        std::string unet_path;
        std::string vae_decoder_path;

        // --- Tokenizer ---
        std::string vocab_path;
        std::string merges_path;

        // --- Generation Parameters ---
        int height = 512;
        int width = 512;
        int num_inference_steps = 20; // 20-50 for DDIM/Euler
        float guidance_scale = 7.5f;  // Classifier-Free Guidance strength
        int seed = -1;              // -1 for random

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    /**
     * @brief Callback for monitoring generation progress.
     * @param step The current step number.
     * @param total_steps Total number of steps.
     * @param latents The current state of the latents (for preview).
     */
    using ProgressCallback = std::function<void(int step, int total_steps, const core::Tensor& latents)>;

    class DiffusionPipeline {
    public:
        explicit DiffusionPipeline(const DiffusionConfig& config);
        ~DiffusionPipeline();

        // Move semantics
        DiffusionPipeline(DiffusionPipeline&&) noexcept;
        DiffusionPipeline& operator=(DiffusionPipeline&&) noexcept;
        DiffusionPipeline(const DiffusionPipeline&) = delete;
        DiffusionPipeline& operator=(const DiffusionPipeline&) = delete;

        /**
         * @brief Generate an image from a text prompt.
         *
         * @param prompt The positive prompt.
         * @param negative_prompt A prompt describing what to avoid.
         * @param progress_callback Optional function to receive updates.
         * @return The final generated image (BGR, uint8).
         */
        cv::Mat generate(const std::string& prompt,
                         const std::string& negative_prompt = "",
                         ProgressCallback progress_callback = nullptr);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative