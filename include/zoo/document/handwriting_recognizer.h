#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::document {

    struct HandwritingRecognitionResult {
        std::string text;
        float confidence;
    };

    struct HandwritingRecognizerConfig {
        std::string engine_path;
        std::string character_map_path;
        int input_height = 64; // Fixed height for CRNN-style recognizer
    };

    class HandwritingRecognizer {
    public:
        explicit HandwritingRecognizer(const HandwritingRecognizerConfig& config);
        ~HandwritingRecognizer();

        HandwritingRecognizer(const HandwritingRecognizer&) = delete;
        HandwritingRecognizer& operator=(const HandwritingRecognizer&) = delete;
        HandwritingRecognizer(HandwritingRecognizer&&) noexcept;
        HandwritingRecognizer& operator=(HandwritingRecognizer&&) noexcept;

        HandwritingRecognitionResult predict(const cv::Mat& line_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document

