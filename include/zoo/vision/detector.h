#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declarations for PIMPL
namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct BoundingBox {
        float x1, y1, x2, y2; // Top-left and bottom-right coordinates
        float confidence;
        int class_id;
        std::string label;
    };

    struct DetectorConfig {
        std::string engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.45f;
        float nms_iou_threshold = 0.5f;
        int input_width = 640;
        int input_height = 640;
    };

    class ObjectDetector {
    public:
        explicit ObjectDetector(const DetectorConfig& config);
        ~ObjectDetector();
        ObjectDetector(const ObjectDetector&) = delete;
        ObjectDetector& operator=(const ObjectDetector&) = delete;
        ObjectDetector(ObjectDetector&&) noexcept;
        ObjectDetector& operator=(ObjectDetector&&) noexcept;

        std::vector<BoundingBox> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

