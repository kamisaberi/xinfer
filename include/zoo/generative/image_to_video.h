#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct VideoGenConfig {
        // Hardware Target (High-end GPU with >16GB VRAM needed)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        std::string image_encoder_path; // VAE Encoder
        std::string video_unet_path;    // 3D UNet
        std::string vae_decoder_path;   // Standard VAE Decoder

        // --- Generation Parameters ---
        int height = 576;
        int width = 1024; // SVD standard
        int num_frames = 14;  // Number of frames to generate
        int motion_bucket_id = 127; // Controls amount of motion
        float fps = 7.0f;
        int num_inference_steps = 25;
        int seed = -1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ImageToVideo {
    public:
        explicit ImageToVideo(const VideoGenConfig& config);
        ~ImageToVideo();

        // Move semantics
        ImageToVideo(ImageToVideo&&) noexcept;
        ImageTo-Video& operator=(ImageToVideo&&) noexcept;
        ImageToVideo(const ImageToVideo&) = delete;
        ImageToVideo& operator=(const ImageToVideo&) = delete;

        /**
         * @brief Generate a video from a starting image.
         *
         * @param init_image The first frame of the video.
         * @return Vector of cv::Mat frames representing the generated clip.
         */
        std::vector<cv::Mat> generate(const cv::Mat& init_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative