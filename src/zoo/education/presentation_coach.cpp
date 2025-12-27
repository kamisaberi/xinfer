#include <xinfer/zoo/education/presentation_coach.h>
#include <xinfer/core/logging.h>

// --- We compose other Zoo modules ---
#include <xinfer/zoo/vision/pose_estimator.h>
#include <xinfer/zoo/hci/emotion_recognizer.h>
#include <xinfer/zoo/audio/speech_recognizer.h>

#include <iostream>
#include <numeric>
#include <string>

namespace xinfer::zoo::education {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PresentationCoach::Impl {
    CoachConfig config_;

    // --- Components ---
    std::unique_ptr<vision::PoseEstimator> pose_estimator_;
    std::unique_ptr<hci::EmotionRecognizer> emotion_recognizer_;
    std::unique_ptr<audio::SpeechRecognizer> speech_recognizer_;

    // --- State ---
    int frame_counter_ = 0;

    // Vocal analysis state
    std::string full_transcript_;
    double total_time_sec_ = 0.0;

    // Feedback smoothing
    std::deque<std::string> emotion_history_;
    int history_size_ = 15; // Smooth over 15 frames

    Impl(const CoachConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init Pose Estimator
        vision::PoseEstimatorConfig pose_cfg;
        pose_cfg.target = config_.target;
        pose_cfg.model_path = config_.pose_model_path;
        pose_estimator_ = std::make_unique<vision::PoseEstimator>(pose_cfg);

        // 2. Init Emotion Recognizer
        hci::EmotionConfig emotion_cfg;
        emotion_cfg.target = config_.target;
        emotion_cfg.detector_path = config_.face_detector_path;
        emotion_cfg.classifier_path = config_.emotion_model_path;
        emotion_cfg.labels = {"Angry", "Happy", "Sad", "Surprise", "Neutral"}; // Example
        emotion_recognizer_ = std::make_unique<hci::EmotionRecognizer>(emotion_cfg);

        // 3. Init Speech Recognizer
        audio::SpeechRecognizerConfig asr_cfg;
        asr_cfg.target = config_.target;
        asr_cfg.model_path = config_.asr_model_path;
        asr_cfg.vocab_path = config_.asr_vocab_path; // ASR model vocab
        // We'd use a generic pre-trained ASR model here
        speech_recognizer_ = std::make_unique<audio::SpeechRecognizer>(asr_cfg);
    }

    // --- Core Logic: Analyze Pose ---
    void analyze_posture(const std::vector<vision::PoseResult>& poses, CoachFeedback& feedback) {
        if (poses.empty()) return;

        const auto& pose = poses[0]; // Assume single person

        // Slouching Heuristic: If head is significantly forward of shoulders
        const auto& nose = pose.keypoints[0];
        const auto& l_shoulder = pose.keypoints[5];
        const auto& r_shoulder = pose.keypoints[6];

        if (l_shoulder.confidence > 0.5 && r_shoulder.confidence > 0.5) {
            float shoulder_mid_y = (l_shoulder.y + r_shoulder.y) / 2.0f;
            float shoulder_height = std::abs(l_shoulder.y - r_shoulder.y);

            // If nose is below shoulder mid-point (looking down)
            if (nose.confidence > 0.5 && nose.y > shoulder_mid_y + shoulder_height) {
                feedback.is_slouching = true;
            } else {
                feedback.has_good_posture = true;
            }
        }

        // Hand Gesture Heuristic: Count how many hands are above waist
        // (Assuming waist is ~ mid-point between shoulders and hips)
        // ... (logic omitted for brevity) ...
    }
};

// =================================================================================
// Public API
// =================================================================================

PresentationCoach::PresentationCoach(const CoachConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PresentationCoach::~PresentationCoach() = default;
PresentationCoach::PresentationCoach(PresentationCoach&&) noexcept = default;
PresentationCoach& PresentationCoach::operator=(PresentationCoach&&) noexcept = default;

void PresentationCoach::reset() {
    if (pimpl_) {
        pimpl_->frame_counter_ = 0;
        pimpl_->full_transcript_ = "";
        pimpl_->total_time_sec_ = 0.0;
        pimpl_->emotion_history_.clear();
    }
}

CoachFeedback PresentationCoach::analyze_frame(const cv::Mat& image, const std::vector<float>& pcm_chunk) {
    if (!pimpl_) throw std::runtime_error("PresentationCoach is null.");

    pimpl_->frame_counter_++;
    CoachFeedback feedback;

    // Run analysis periodically, not every single frame
    if (pimpl_->frame_counter_ % pimpl_->config_.analysis_interval != 0) {
        return feedback; // Return empty feedback
    }

    // 1. Posture Analysis
    auto poses = pimpl_->pose_estimator_->estimate(image);
    pimpl_->analyze_posture(poses, feedback);

    // 2. Emotion Analysis
    auto emotions = pimpl_->emotion_recognizer_->recognize(image);
    if (!emotions.empty()) {
        // Smoothing
        pimpl_->emotion_history_.push_back(emotions[0].second.dominant_emotion);
        if (pimpl_->emotion_history_.size() > pimpl_->config_.analysis_interval) {
            pimpl_->emotion_history_.pop_front();
        }

        // Find most common emotion in history
        // (Logic for finding mode of deque omitted for brevity)
        feedback.dominant_emotion = pimpl_->emotion_history_.back();
    }

    // 3. Vocal Analysis
    // We run ASR on the new chunk and append to transcript
    if (!pcm_chunk.empty()) {
        auto asr_res = pimpl_->speech_recognizer_->recognize(pcm_chunk);
        if (!asr_res.empty()) {
            pimpl_->full_transcript_ += asr_res[0] + " ";
        }
    }

    pimpl_->total_time_sec_ += (float)pimpl_->config_.analysis_interval / pimpl_->config_.video_fps;

    // Calculate WPM
    if (pimpl_->total_time_sec_ > 1.0) {
        std::stringstream ss(pimpl_->full_transcript_);
        std::string word;
        int word_count = 0;
        while (ss >> word) word_count++;

        feedback.words_per_minute = (float)word_count / (pimpl_->total_time_sec_ / 60.0);
    }

    // Filler word count (simple string search)
    // In a real app, use a proper list and regex
    feedback.filler_word_count = 0;
    // ... (logic to find "um", "ah", "like" in full_transcript_) ...

    // 4. Generate Feedback
    if (feedback.words_per_minute > 160) {
        feedback.primary_feedback = "Pace is a bit fast. Try to slow down.";
    } else if (feedback.words_per_minute < 110 && feedback.words_per_minute > 10) {
        feedback.primary_feedback = "Good pace! Keep it up.";
    } else {
        feedback.primary_feedback = "Looking good!";
    }

    if (feedback.is_slouching) {
        feedback.primary_feedback += " Try to keep your posture upright.";
    }

    return feedback;
}

} // namespace xinfer::zoo::education```