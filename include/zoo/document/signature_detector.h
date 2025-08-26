#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::document {

    struct Signature {
        float confidence;
        cv::Rect bounding_box;
    };

    struct SignatureDetectorConfig {
        std::string engine_path;
        float confidence_threshold = 0.6f;
        float nms_iou_threshold = 0.4f;
        int input_width = 640;
        int input_height = 640;
    };

    class SignatureDetector {
    public:
        explicit SignatureDetector(const SignatureDetectorConfig& config);
        ~SignatureDetector();

        SignatureDetector(const SignatureDetector&) = delete;
        SignatureDetector& operator=(const SignatureDetector&) = delete;
        SignatureDetector(SignatureDetector&&) noexcept;
        SignatureDetector& operator=(SignatureDetector&&) noexcept;

        std::vector<Signature> predict(const cv::Mat& document_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document

