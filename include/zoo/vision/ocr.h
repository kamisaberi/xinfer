#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Forward declarations for PIMPL
namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct OCRResult {
        std::vector<cv::Point2f> box_points;
        std::string text;
        float confidence;
    };

    struct OCRConfig {
        std::string detection_engine_path;
        std::string recognition_engine_path;
        float text_threshold = 0.7f;
        float box_threshold = 0.4f; // For CRAFT decoder
        int recognition_input_height = 32;
        std::string character_set = "0123456789abcdefghijklmnopqrstuvwxyz";
    };

    class OCR {
    public:
        explicit OCR(const OCRConfig& config);
        ~OCR();
        OCR(const OCR&) = delete;
        OCR& operator=(const OCR&) = delete;
        OCR(OCR&&) noexcept;
        OCR& operator=(OCR&&) noexcept;

        std::vector<OCRResult> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

