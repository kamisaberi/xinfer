#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct ColorizerConfig {
        std::string engine_path;
        int input_width = 256;
        int input_height = 256;
    };

    class Colorizer {
    public:
        explicit Colorizer(const ColorizerConfig& config);
        ~Colorizer();

        Colorizer(const Colorizer&) = delete;
        Colorizer& operator=(const Colorizer&) = delete;
        Colorizer(Colorizer&&) noexcept;
        Colorizer& operator=(Colorizer&&) noexcept;

        cv::Mat predict(const cv::Mat& grayscale_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

