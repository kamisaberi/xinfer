#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::geospatial {

    struct DetectedObject {
        int class_id;
        float confidence;
        std::string label;
        std::vector<cv::Point2f> contour; // For rotated bounding boxes
    };

    struct MaritimeDetectorConfig {
        std::string engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.5f;
        float nms_iou_threshold = 0.4f;
        int input_width = 1024;
        int input_height = 1024;
    };

    class MaritimeDetector {
    public:
        explicit MaritimeDetector(const MaritimeDetectorConfig& config);
        ~MaritimeDetector();

        MaritimeDetector(const MaritimeDetector&) = delete;
        MaritimeDetector& operator=(const MaritimeDetector&) = delete;
        MaritimeDetector(MaritimeDetector&&) noexcept;
        MaritimeDetector& operator=(MaritimeDetector&&) noexcept;

        std::vector<DetectedObject> predict(const cv::Mat& satellite_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial

