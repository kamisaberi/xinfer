#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct InpainterConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class Inpainter {
    public:
        explicit Inpainter(const InpainterConfig& config);
        ~Inpainter();

        Inpainter(const Inpainter&) = delete;
        Inpainter& operator=(const Inpainter&) = delete;
        Inpainter(Inpainter&&) noexcept;
        Inpainter& operator=(Inpainter&&) noexcept;

        cv::Mat predict(const cv::Mat& image, const cv::Mat& mask);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

