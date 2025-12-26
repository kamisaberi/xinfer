#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct SuperResolutionConfig {
        // Hardware Target (GPU is recommended for high-res upscaling)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., realesrgan_x4.engine)
        std::string model_path;

        // Model Scale
        // e.g., 2 for 2x, 4 for 4x
        int scale = 4;

        // Input Specs (Can be variable size, but tiling is often used for large images)
        // Some models require specific input patch sizes.
        int input_width = 0;  // 0 = auto/dynamic
        int input_height = 0; // 0 = auto/dynamic

        // Tiling (For very large images to manage VRAM)
        int tile_size = 512;
        int tile_pad = 10;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SuperResolution {
    public:
        explicit SuperResolution(const SuperResolutionConfig& config);
        ~SuperResolution();

        // Move semantics
        SuperResolution(SuperResolution&&) noexcept;
        SuperResolution& operator=(SuperResolution&&) noexcept;
        SuperResolution(const SuperResolution&) = delete;
        SuperResolution& operator=(const SuperResolution&) = delete;

        /**
         * @brief Upscale an image.
         *
         * @param lr_image The low-resolution input image (BGR).
         * @return The high-resolution output image.
         */
        cv::Mat upscale(const cv::Mat& lr_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative