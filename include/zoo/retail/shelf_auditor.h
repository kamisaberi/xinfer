#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::vision { class ObjectDetector; }

namespace xinfer::zoo::retail {

    struct ShelfItem {
        int class_id;
        std::string label;
        int count;
        std::vector<cv::Rect> locations;
    };

    struct ShelfAuditorConfig {
        std::string detection_engine_path;
        std::string labels_path;
        float confidence_threshold = 0.6f;
        float nms_iou_threshold = 0.4f;
        int detection_input_width = 1280;
        int detection_input_height = 720;
    };

    class ShelfAuditor {
    public:
        explicit ShelfAuditor(const ShelfAuditorConfig& config);
        ~ShelfAuditor();

        ShelfAuditor(const ShelfAuditor&) = delete;
        ShelfAuditor& operator=(const ShelfAuditor&) = delete;
        ShelfAuditor(ShelfAuditor&&) noexcept;
        ShelfAuditor& operator=(ShelfAuditor&&) noexcept;

        std::vector<ShelfItem> audit(const cv::Mat& shelf_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail

