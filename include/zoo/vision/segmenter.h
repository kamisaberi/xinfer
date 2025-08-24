#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct SegmenterConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class Segmenter {
    public:
        explicit Segmenter(const SegmenterConfig& config);
        ~Segmenter();

        Segmenter(const Segmenter&) = delete;
        Segmenter& operator=(const Segmenter&) = delete;
        Segmenter(Segmenter&&) noexcept;
        Segmenter& operator=(Segmenter&&) noexcept;

        cv::Mat predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

