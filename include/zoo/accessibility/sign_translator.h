#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For Keypoint struct from Pose

namespace xinfer::zoo::accessibility {

    struct SignResult {
        std::string current_word;
        float confidence;
        bool is_new_word; // True if a new word was just completed
    };

    struct SignConfig {
        // Hardware Target (Real-time requires GPU/NPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Pose/Hand Keypoint Detector ---
        std::string pose_model_path;

        // --- Model 2: Sign Classifier (Sequence Model) ---
        // Takes a sequence of keypoints as input
        std::string classifier_model_path;
        std::string labels_path; // "Hello", "Thank You", "A", "B", ...

        // --- Logic ---
        int sequence_length = 30; // Model expects 30 frames of keypoints
        float min_confidence = 0.7f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SignTranslator {
    public:
        explicit SignTranslator(const SignConfig& config);
        ~SignTranslator();

        // Move semantics
        SignTranslator(SignTranslator&&) noexcept;
        SignTranslator& operator=(SignTranslator&&) noexcept;
        SignTranslator(const SignTranslator&) = delete;
        SignTranslator& operator=(const SignTranslator&) = delete;

        /**
         * @brief Process a video frame to translate signs.
         *
         * @param image Input camera frame.
         * @return The currently detected word or phrase.
         */
        SignResult translate(const cv::Mat& image);

        /**
         * @brief Reset the sequence buffer.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::accessibility