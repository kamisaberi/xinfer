#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct ImageDeblurConfig {
        std::string engine_path;
        int input_width = 256;
        int input_height = 256;
    };

    class ImageDeblur {
    public:
        explicit ImageDeblur(const ImageDeblurConfig& config);
        ~ImageDeblur();

        ImageDeblur(const ImageDeblur&) = delete;
        ImageDeblur& operator=(const ImageDeblur&) = delete;
        ImageDeblur(ImageDeblur&&) noexcept;
        ImageDeblur& operator=(ImageDeblur&&) noexcept;

        cv::Mat predict(const cv::Mat& blurry_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

