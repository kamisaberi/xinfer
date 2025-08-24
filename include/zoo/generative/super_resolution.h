#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct SuperResolutionConfig {
        std::string engine_path;
        int upscale_factor = 4;
        int input_width = 128;
        int input_height = 128;
    };

    class SuperResolution {
    public:
        explicit SuperResolution(const SuperResolutionConfig& config);
        ~SuperResolution();

        SuperResolution(const SuperResolution&) = delete;
        SuperResolution& operator=(const SuperResolution&) = delete;
        SuperResolution(SuperResolution&&) noexcept;
        SuperResolution& operator=(SuperResolution&&) noexcept;

        cv::Mat predict(const cv::Mat& low_res_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

