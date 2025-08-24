#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct ChangeDetectorConfig {
        std::string engine_path;
        float change_threshold = 0.5f;
        int input_width = 512;
        int input_height = 512;
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};
    };

    class ChangeDetector {
    public:
        explicit ChangeDetector(const ChangeDetectorConfig& config);
        ~ChangeDetector();

        ChangeDetector(const ChangeDetector&) = delete;
        ChangeDetector& operator=(const ChangeDetector&) = delete;
        ChangeDetector(ChangeDetector&&) noexcept;
        ChangeDetector& operator=(ChangeDetector&&) noexcept;

        cv::Mat predict(const cv::Mat& image_before, const cv::Mat& image_after);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

