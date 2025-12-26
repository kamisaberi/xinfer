#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct InpainterConfig {
        // Hardware Target (GPU with >8GB VRAM is required)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        std::string text_encoder_path;
        std::string unet_path; // MUST be an Inpainting UNet
        std::string vae_encoder_path;
        std::string vae_decoder_path;

        // --- Tokenizer ---
        std::string vocab_path;
        std::string merges_path;

        // --- Generation Parameters ---
        int height = 512;
        int width = 512;
        int num_inference_steps = 50;
        float guidance_scale = 7.5f;
        int seed = -1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Inpainter {
    public:
        explicit Inpainter(const InpainterConfig& config);
        ~Inpainter();

        // Move semantics
        Inpainter(Inpainter&&) noexcept;
        Inpainter& operator=(Inpainter&&) noexcept;
        Inpainter(const Inpainter&) = delete;
        Inpainter& operator=(const Inpainter&) = delete;

        /**
         * @brief Inpaint a masked region of an image.
         *
         * @param image The original image (BGR).
         * @param mask The mask image (Grayscale, 255=Inpaint, 0=Keep).
         * @param prompt Text prompt describing what to fill in.
         * @param negative_prompt Optional negative prompt.
         * @return The completed image.
         */
        cv::Mat inpaint(const cv::Mat& image,
                        const cv::Mat& mask,
                        const std::string& prompt,
                        const std::string& negative_prompt = "");

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative