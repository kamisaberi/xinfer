#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct OutpainterConfig {
        // Hardware Target (GPU required)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // Uses the same models as Inpainting
        std::string text_encoder_path;
        std::string unet_path; // MUST be an Inpainting UNet
        std::string vae_encoder_path;
        std::string vae_decoder_path;

        // --- Tokenizer ---
        std::string vocab_path;
        std::string merges_path;

        // --- Generation Parameters ---
        int base_height = 512; // Size of the model's native generation space
        int base_width = 512;
        int num_inference_steps = 50;
        float guidance_scale = 7.5f;
        int seed = -1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Outpainter {
    public:
        explicit Outpainter(const OutpainterConfig& config);
        ~Outpainter();

        // Move semantics
        Outpainter(Outpainter&&) noexcept;
        Outpainter& operator=(Outpainter&&) noexcept;
        Outpainter(const Outpainter&) = delete;
        Outpainter& operator=(const Outpainter&) = delete;

        /**
         * @brief Extend an image.
         *
         * @param image The original image (BGR).
         * @param top Pixels to add on top.
         * @param bottom Pixels to add on bottom.
         * @param left Pixels to add on left.
         * @param right Pixels to add on right.
         * @param prompt Text prompt to guide the new content.
         * @return The expanded image.
         */
        cv::Mat outpaint(const cv::Mat& image,
                         int top, int bottom, int left, int right,
                         const std::string& prompt);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative