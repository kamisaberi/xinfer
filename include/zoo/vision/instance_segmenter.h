#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct InstanceSegmentationResult {
        int class_id;
        float confidence;
        std::string label;
        cv::Rect bounding_box;
        cv::Mat mask;
    };

    struct InstanceSegmenterConfig {
        std::string engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.5f;
        float nms_iou_threshold = 0.5f;
        float mask_threshold = 0.5f;
        int input_width = 640;
        int input_height = 640;
    };

    class InstanceSegmenter {
    public:
        explicit InstanceSegmenter(const InstanceSegmenterConfig& config);
        ~InstanceSegmenter();

        InstanceSegmenter(const InstanceSegmenter&) = delete;
        InstanceSegmenter& operator=(const InstanceSegmenter&) = delete;
        InstanceSegmenter(InstanceSegmenter&&) noexcept;
        InstanceSegmenter& operator=(InstanceSegmenter&&) noexcept;

        std::vector<InstanceSegmentationResult> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

