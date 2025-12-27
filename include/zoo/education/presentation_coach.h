#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::education {

    struct CoachFeedback {
        // Body Language
        bool is_slouching = false;
        bool has_good_posture = false;
        int hand_gesture_count = 0;

        // Facial Expression
        std::string dominant_emotion; // "Neutral", "Happy", "Anxious"

        // Vocal Delivery
        float words_per_minute;
        int filler_word_count;

        // Overall
        std::string primary_feedback; // e.g., "Good energy!", "Try to slow down."
    };

    struct CoachConfig {
        // Hardware Target (CPU is often fine for single stream)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Models ---
        // 1. Pose Estimator (e.g., YOLOv8-Pose)
        std::string pose_model_path;

        // 2. Emotion Classifier
        std::string emotion_model_path;
        std::string face_detector_path; // For emotion classifier

        // 3. Speech Recognizer (for WPM and filler words)
        std::string asr_model_path;
        std::string asr_vocab_path;

        // --- Logic ---
        float video_fps = 30.0f;
        int analysis_interval = 15; // Run analysis every 15 frames

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PresentationCoach {
    public:
        explicit PresentationCoach(const CoachConfig& config);
        ~PresentationCoach();

        // Move semantics
        PresentationCoach(PresentationCoach&&) noexcept;
        PresentationCoach& operator=(Presentation-Coach&&) noexcept;
        PresentationCoach(const PresentationCoach&) = delete;
        PresentationCoach& operator=(const PresentationCoach&) = delete;

        /**
         * @brief Process a frame of the presentation video and its corresponding audio chunk.
         *
         * @param image The video frame.
         * @param pcm_chunk The audio samples for this frame's duration.
         * @return Feedback struct (updated periodically).
         */
        CoachFeedback analyze_frame(const cv::Mat& image, const std::vector<float>& pcm_chunk);

        /**
         * @brief Reset all internal metrics (e.g. for a new presentation).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::education