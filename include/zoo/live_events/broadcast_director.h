#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::live_events {

    enum class CameraShot {
        WIDE_TACTICAL = 0, // Show full formation
        MID_FOLLOW_BALL = 1, // Standard broadcast view
        CLOSE_UP_PLAYER = 2, // Highlight key player
        REPLAY_ACTION = 3    // Trigger replay of a significant event
    };

    struct BroadcastDecision {
        CameraShot recommended_shot;
        int target_player_id; // -1 if not a close-up
        float shot_confidence;
        std::string event_detected; // "Goal", "Shot on Target", "Pass"
    };

    struct DirectorConfig {
        // Hardware Target (High-end GPU required for multi-model, high-FPS)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Player/Ball Detector (YOLO) ---
        std::string detector_path;
        int det_input_width = 1280;
        int det_input_height = 720;
        float det_conf_thresh = 0.4f;

        // --- Model 2: Action Classifier (Video 3D-CNN) ---
        std::string action_model_path;
        int action_window_size = 16; // Needs 16 frames of history

        // --- Class Mappings ---
        int player_class_id = 0;
        int ball_class_id = 1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class BroadcastDirector {
    public:
        explicit BroadcastDirector(const DirectorConfig& config);
        ~BroadcastDirector();

        // Move semantics
        BroadcastDirector(BroadcastDirector&&) noexcept;
        BroadcastDirector& operator=(BroadcastDirector&&) noexcept;
        BroadcastDirector(const BroadcastDirector&) = delete;
        BroadcastDirector& operator=(const BroadcastDirector&) = delete;

        /**
         * @brief Analyze a live broadcast frame and make a directorial decision.
         *
         * @param image Input video frame.
         * @return The recommended camera action.
         */
        BroadcastDecision direct_frame(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::live_events