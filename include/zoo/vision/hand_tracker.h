#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct Hand {
        int track_id;
        float confidence;
        float x1, y1, x2, y2;
        std::vector<cv::Point2f> keypoints;
    };

    struct HandTrackerConfig {
        std::string detection_engine_path;
        float confidence_threshold = 0.6f;
        float nms_iou_threshold = 0.5f;
        int input_width = 256;
        int input_height = 256;
    };

    class HandTracker {
    public:
        explicit HandTracker(const HandTrackerConfig& config);
        ~HandTracker();

        HandTracker(const HandTracker&) = delete;
        HandTracker& operator=(const HandTracker&) = delete;
        HandTracker(HandTracker&&) noexcept;
        HandTracker& operator=(HandTracker&&) noexcept;

        std::vector<Hand> track(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

