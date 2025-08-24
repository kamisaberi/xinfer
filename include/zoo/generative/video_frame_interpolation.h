#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::generative {

    struct VideoFrameInterpolationConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class VideoFrameInterpolation {
    public:
        explicit VideoFrameInterpolation(const VideoFrameInterpolationConfig& config);
        ~VideoFrameInterpolation();

        VideoFrameInterpolation(const VideoFrameInterpolation&) = delete;
        VideoFrameInterpolation& operator=(const VideoFrameInterpolation&) = delete;
        VideoFrameInterpolation(VideoFrameInterpolation&&) noexcept;
        VideoFrameInterpolation& operator=(VideoFrameInterpolation&&) noexcept;

        cv::Mat predict(const cv::Mat& frame_before, const cv::Mat& frame_after);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

