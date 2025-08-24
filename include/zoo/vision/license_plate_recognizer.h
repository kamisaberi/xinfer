#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct LPResult {
        std::string text;
        float confidence;
        std::vector<cv::Point2f> box_points; // Corners of the license plate
    };

    struct LicensePlateRecognizerConfig {
        std::string detection_engine_path;
        std::string recognition_engine_path;
        float detection_confidence_threshold = 0.5f;
        float detection_nms_iou_threshold = 0.4f;
        int detection_input_width = 640;
        int detection_input_height = 480;
        int recognition_input_height = 32; // Fixed height for CRNN-style recognizer
        std::string character_set = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"; // Common alphanumeric set
    };

    class LicensePlateRecognizer {
    public:
        explicit LicensePlateRecognizer(const LicensePlateRecognizerConfig& config);
        ~LicensePlateRecognizer();

        LicensePlateRecognizer(const LicensePlateRecognizer&) = delete;
        LicensePlateRecognizer& operator=(const LicensePlateRecognizer&) = delete;
        LicensePlateRecognizer(LicensePlateRecognizer&&) noexcept;
        LicensePlateRecognizer& operator=(LicensePlateRecognizer&&) noexcept;

        std::vector<LPResult> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

