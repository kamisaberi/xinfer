#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct OpticalFlowConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 384;
    };

    class OpticalFlow {
    public:
        explicit OpticalFlow(const OpticalFlowConfig& config);
        ~OpticalFlow();

        OpticalFlow(const OpticalFlow&) = delete;
        OpticalFlow& operator=(const OpticalFlow&) = delete;
        OpticalFlow(OpticalFlow&&) noexcept;
        OpticalFlow& operator=(OpticalFlow&&) noexcept;

        cv::Mat predict(const cv::Mat& frame1, const cv::Mat& frame2);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

