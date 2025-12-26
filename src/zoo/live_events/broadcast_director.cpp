#include <xinfer/zoo/live_events/broadcast_director.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::live_events {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct BroadcastDirector::Impl {
    DirectorConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    std::unique_ptr<postproc::ITracker> tracker_;

    std::unique_ptr<backends::IBackend> action_engine_;
    // We reuse the Video preprocessor (Frame Stacker)
    // std::unique_ptr<preproc::IVideoPreprocessor> action_preproc_;
    // And classification postprocessor
    std::unique_ptr<postproc::IClassificationPostprocessor> action_postproc_;

    // Data Containers
    core::Tensor det_input, det_output;
    core::Tensor action_input, action_output;

    // State
    CameraShot current_shot_ = CameraShot::WIDE_TACTICAL;

    Impl(const DirectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.detector_path;
        d_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("BroadcastDirector: Failed to load detector.");
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.det_input_width;
        dp_cfg.target_height = config_.det_input_height;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = config_.det_conf_thresh;
        det_postproc_->init(dpost_cfg);

        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        tracker_->init(trk_cfg);

        // 2. Setup Action Recognizer
        // ... (Initialization logic for 3D CNN model and Frame Stacker preproc) ...
    }

    // --- Director Logic (State Machine) ---
    BroadcastDecision make_decision(const std::vector<postproc::TrackedObject>& tracks) {
        BroadcastDecision decision;
        decision.recommended_shot = current_shot_;
        decision.target_player_id = -1;

        // Find the ball
        const postproc::TrackedObject* ball = nullptr;
        for (const auto& t : tracks) {
            if (t.box.class_id == config_.ball_class_id) {
                ball = &t;
                break;
            }
        }

        if (!ball) {
            // Ball is out of play or occluded, stay on wide shot
            decision.recommended_shot = CameraShot::WIDE_TACTICAL;
            current_shot_ = decision.recommended_shot;
            return decision;
        }

        // Find closest player to the ball
        float min_dist = 1e9;
        int player_with_ball_id = -1;

        for (const auto& t : tracks) {
            if (t.box.class_id == config_.player_class_id) {
                float dx = (t.box.x1 + t.box.x2)/2 - (ball->box.x1 + ball->box.x2)/2;
                float dy = (t.box.y1 + t.box.y2)/2 - (ball->box.y1 + ball->box.y2)/2;
                float dist = dx*dx + dy*dy;
                if (dist < min_dist) {
                    min_dist = dist;
                    player_with_ball_id = t.track_id;
                }
            }
        }

        // Rule-Based Decision
        // Get ball X position, normalized [0-1]
        float ball_x = ((ball->box.x1 + ball->box.x2) / 2.0f) / config_.det_input_width;

        if (ball_x > 0.3 && ball_x < 0.7) {
            // Midfield play -> Standard follow
            decision.recommended_shot = CameraShot::MID_FOLLOW_BALL;
        } else {
            // In the attacking third -> Close up on player
            decision.recommended_shot = CameraShot::CLOSE_UP_PLAYER;
            decision.target_player_id = player_with_ball_id;
        }

        // Action recognition logic would go here
        // if(action_result == "Goal") decision.recommended_shot = REPLAY_ACTION;

        current_shot_ = decision.recommended_shot;
        return decision;
    }
};

// =================================================================================
// Public API
// =================================================================================

BroadcastDirector::BroadcastDirector(const DirectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

BroadcastDirector::~BroadcastDirector() = default;
BroadcastDirector::BroadcastDirector(BroadcastDirector&&) noexcept = default;
BroadcastDirector& BroadcastDirector::operator=(BroadcastDirector&&) noexcept = default;

BroadcastDecision BroadcastDirector::direct_frame(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("BroadcastDirector is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);

    // 2. Inference
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});

    // 3. Detect
    auto raw_dets = pimpl_->det_postproc_->process({pimpl_->det_output});

    // 4. Track
    // No coordinate scaling needed if we do logic in model space
    auto tracks = pimpl_->tracker_->update(raw_dets);

    // 5. Decision Logic
    return pimpl_->make_decision(tracks);
}

} // namespace xinfer::zoo::live_events