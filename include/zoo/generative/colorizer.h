#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct ColorizerConfig {
        // Hardware Target (Generative models run best on GPU/NPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., colorization_gan.engine)
        // Expected Input: [1, 1, H, W] (L Channel)
        // Expected Output: [1, 2, H, W] (a, b Channels)
        std::string model_path;

        // Input Specs
        int input_width = 256;
        int input_height = 256;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Colorizer {
    public:
        explicit Colorizer(const ColorizerConfig& config);
        ~Colorizer();

        // Move semantics
        Colorizer(Colorizer&&) noexcept;
        Colorizer& operator=(Colorizer&&) noexcept;
        Colorizer(const Colorizer&) = delete;
        Colorizer& operator=(const Colorizer&) = delete;

        /**
         * @brief Colorize a grayscale image.
         *
         * @param gray_image Input image (can be CV_8UC1 or CV_8UC3, will be converted).
         * @return The colorized image in BGR format.
         */
        cv::Mat colorize(const cv::Mat& gray_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative