#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct StyleTransferConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class StyleTransfer {
    public:
        explicit StyleTransfer(const StyleTransferConfig& config);
        ~StyleTransfer();

        StyleTransfer(const StyleTransfer&) = delete;
        StyleTransfer& operator=(const StyleTransfer&) = delete;
        StyleTransfer(StyleTransfer&&) noexcept;
        StyleTransfer& operator=(StyleTransfer&&) noexcept;

        cv::Mat predict(const cv::Mat& content_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

