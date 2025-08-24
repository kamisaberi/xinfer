#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct Face {
        float confidence;
        float x1, y1, x2, y2;
        std::vector<cv::Point2f> landmarks;
    };

    struct FaceDetectorConfig {
        std::string engine_path;
        float confidence_threshold = 0.7f;
        float nms_iou_threshold = 0.4f;
        int input_width = 640;
        int input_height = 480;
    };

    class FaceDetector {
    public:
        explicit FaceDetector(const FaceDetectorConfig& config);
        ~FaceDetector();

        FaceDetector(const FaceDetector&) = delete;
        FaceDetector& operator=(const FaceDetector&) = delete;
        FaceDetector(FaceDetector&&) noexcept;
        FaceDetector& operator=(FaceDetector&&) noexcept;

        std::vector<Face> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

