#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/core/tensor.h>

namespace xinfer::zoo::generative {

    struct ImageToVideoConfig {
        std::string engine_path;
        int num_frames = 16;
        int input_width = 256;
        int input_height = 256;
    };

    class ImageToVideo {
    public:
        explicit ImageToVideo(const ImageToVideoConfig& config);
        ~ImageToVideo();

        ImageToVideo(const ImageToVideo&) = delete;
        ImageToVideo& operator=(const ImageToVideo&) = delete;
        ImageToVideo(ImageToVideo&&) noexcept;
        ImageToVideo& operator=(ImageToVideo&&) noexcept;

        std::vector<cv::Mat> predict(const cv::Mat& start_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

