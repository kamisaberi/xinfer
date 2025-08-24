#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct OutpainterConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class Outpainter {
    public:
        explicit Outpainter(const OutpainterConfig& config);
        ~Outpainter();

        Outpainter(const Outpainter&) = delete;
        Outpainter& operator=(const Outpainter&) = delete;
        Outpainter(Outpainter&&) noexcept;
        Outpainter& operator=(Outpainter&&) noexcept;

        cv::Mat predict(const cv::Mat& image, int top, int right, int bottom, int left);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

