#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct DepthEstimatorConfig {
        std::string engine_path;
        int input_width = 384;
        int input_height = 384;
    };

    class DepthEstimator {
    public:
        explicit DepthEstimator(const DepthEstimatorConfig& config);
        ~DepthEstimator();

        DepthEstimator(const DepthEstimator&) = delete;
        DepthEstimator& operator=(const DepthEstimator&) = delete;
        DepthEstimator(DepthEstimator&&) noexcept;
        DepthEstimator& operator=(DepthEstimator&&) noexcept;

        cv::Mat predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

