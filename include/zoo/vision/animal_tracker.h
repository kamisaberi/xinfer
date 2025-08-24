#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct TrackedAnimal {
        int track_id;
        int class_id;
        std::string label;
        float confidence;
        float x1, y1, x2, y2;
    };

    struct AnimalTrackerConfig {
        std::string detection_engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.5f;
        float nms_iou_threshold = 0.5f;
        int input_width = 640;
        int input_height = 640;
    };

    class AnimalTracker {
    public:
        explicit AnimalTracker(const AnimalTrackerConfig& config);
        ~AnimalTracker();

        AnimalTracker(const AnimalTracker&) = delete;
        AnimalTracker& operator=(const AnimalTracker&) = delete;
        AnimalTracker(AnimalTracker&&) noexcept;
        AnimalTracker& operator=(AnimalTracker&&) noexcept;

        std::vector<TrackedAnimal> track(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

