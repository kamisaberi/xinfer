#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::threed {

    struct SLAMFeatureResult {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
    };

    struct SLAMAcceleratorConfig {
        std::string feature_engine_path;
        int input_width = 640;
        int input_height = 480;
    };

    class SLAMAccelerator {
    public:
        explicit SLAMAccelerator(const SLAMAcceleratorConfig& config);
        ~SLAMAccelerator();

        SLAMAccelerator(const SLAMAccelerator&) = delete;
        SLAMAccelerator& operator=(const SLAMAccelerator&) = delete;
        SLAMAccelerator(SLAMAccelerator&&) noexcept;
        SLAMAccelerator& operator=(SLAMAccelerator&&) noexcept;

        SLAMFeatureResult extract_features(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed

