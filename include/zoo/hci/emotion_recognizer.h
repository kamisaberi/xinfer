#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::hci {

    /**
     * @brief Result of emotion recognition.
     */
    struct EmotionResult {
        std::string dominant_emotion; // e.g., "Happy"
        float confidence;

        // Full breakdown of probabilities
        std::map<std::string, float> scores;
    };

    struct EmotionConfig {
        // Hardware Target (Edge CPU/NPU is common for HCI)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Face Detector (Optional but Recommended) ---
        // If empty, the module will process the full image.
        std::string detector_path;

        // --- Model 2: Emotion Classifier ---
        std::string classifier_path;

        // Input Specs for Classifier (Grayscale, e.g. 48x48 or 64x64)
        int input_width = 64;
        int input_height = 64;

        // Label map for the classifier
        // e.g., ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        std::vector<std::string> labels;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class EmotionRecognizer {
    public:
        explicit EmotionRecognizer(const EmotionConfig& config);
        ~EmotionRecognizer();

        // Move semantics
        EmotionRecognizer(EmotionRecognizer&&) noexcept;
        EmotionRecognizer& operator=(EmotionRecognizer&&) noexcept;
        EmotionRecognizer(const EmotionRecognizer&) = delete;
        EmotionRecognizer& operator=(const EmotionRecognizer&) = delete;

        /**
         * @brief Recognize emotions from all faces in an image.
         *
         * Pipeline:
         * 1. Detect faces.
         * 2. Crop, convert to grayscale, and resize each face.
         * 3. Classify emotion for each face.
         *
         * @param image Input frame.
         * @return List of results, one per detected face.
         */
        std::vector<std::pair<cv::Rect, EmotionResult>> recognize(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::hci