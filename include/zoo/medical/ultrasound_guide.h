#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct UltrasoundGuideResult {
        cv::Mat segmentation_mask;
        std::vector<std::vector<cv::Point>> contours;
    };

    struct UltrasoundGuideConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class UltrasoundGuide {
    public:
        explicit UltrasoundGuide(const UltrasoundGuideConfig& config);
        ~UltrasoundGuide();

        UltrasoundGuide(const UltrasoundGuide&) = delete;
        UltrasoundGuide& operator=(const UltrasoundGuide&) = delete;
        UltrasoundGuide(UltrasoundGuide&&) noexcept;
        UltrasoundGuide& operator=(UltrasoundGuide&&) noexcept;

        UltrasoundGuideResult predict(const cv::Mat& ultrasound_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical

