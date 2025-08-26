#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::document {

    struct LayoutElement {
        int class_id;
        float confidence;
        std::string label;
        cv::Rect bounding_box;
    };

    struct LayoutParserConfig {
        std::string engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.7f;
        float nms_iou_threshold = 0.5f;
        int input_width = 800;
        int input_height = 1024;
    };

    class LayoutParser {
    public:
        explicit LayoutParser(const LayoutParserConfig& config);
        ~LayoutParser();

        LayoutParser(const LayoutParser&) = delete;
        LayoutParser& operator=(const LayoutParser&) = delete;
        LayoutParser(LayoutParser&&) noexcept;
        LayoutParser& operator=(LayoutParser&&) noexcept;

        std::vector<LayoutElement> predict(const cv::Mat& document_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document

