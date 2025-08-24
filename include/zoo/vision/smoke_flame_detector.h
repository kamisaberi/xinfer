#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct DetectionEvent {
        enum class Type { NONE, SMOKE, FLAME };
        Type type = Type::NONE;
        float confidence = 0.0f;
        cv::Rect bounding_box;
    };

    struct SmokeFlameDetectorConfig {
        std::string engine_path;
        float confidence_threshold = 0.6f;
        float nms_iou_threshold = 0.4f;
        int input_width = 640;
        int input_height = 640;
    };

    class SmokeFlameDetector {
    public:
        explicit SmokeFlameDetector(const SmokeFlameDetectorConfig& config);
        ~SmokeFlameDetector();

        SmokeFlameDetector(const SmokeFlameDetector&) = delete;
        SmokeFlameDetector& operator=(const SmokeFlameDetector&) = delete;
        SmokeFlameDetector(SmokeFlameDetector&&) noexcept;
        SmokeFlameDetector& operator=(SmokeFlameDetector&&) noexcept;

        std::vector<DetectionEvent> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

