#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct LowLightEnhancerConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class LowLightEnhancer {
    public:
        explicit LowLightEnhancer(const LowLightEnhancerConfig& config);
        ~LowLightEnhancer();

        LowLightEnhancer(const LowLightEnhancer&) = delete;
        LowLightEnhancer& operator=(const LowLightEnhancer&) = delete;
        LowLightEnhancer(LowLightEnhancer&&) noexcept;
        LowLightEnhancer& operator=(LowLightEnhancer&&) noexcept;

        cv::Mat predict(const cv::Mat& low_light_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

