#include <xinfer/zoo/live_events/sports_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <map>
#include <numeric>

namespace xinfer::zoo::live_events {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SportsAnalyzer::Impl {
    SportsConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // --- Tensors ---
    core::Tensor input_tensor, output_tensor;

    // --- State ---
    struct TrackState {
        int team_id;
        int class_id;
        std::vector<cv::Point2f> history; // For trajectory
    };
    std::map<int, TrackState> track_history_;

    // Possession tracking
    int possession_team_a_frames_ = 0;
    int possession_team_b_frames_ = 0;
    int total_frames_ = 0;

    Impl(const SportsConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config cfg; cfg.model_path = config_.detector_path;

        if (!engine_->load_model(cfg.model_path)) {
            throw std::runtime_error("SportsAnalyzer: Failed to load detector model.");
        }

        // 2. Setup Preproc
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        preproc_->init(pre_cfg);

        // 3. Setup Detector Postproc
        detector_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        detector_->init(det_cfg);

        // 4. Setup Tracker
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        tracker_->init(trk_cfg);
    }

    // --- Helper: Project pixel coords to top-down field coords ---
    cv::Point2f to_field_coords(const cv::Point2f& pixel_pt) {
        if (config_.homography_matrix.empty()) return pixel_pt; // No transform

        std::vector<cv::Point2f> src = {pixel_pt};
        std::vector<cv::Point2f> dst;
        cv::perspectiveTransform(src, dst, config_.homography_matrix);
        return dst[0];
    }
};

// =================================================================================
// Public API
// =================================================================================

SportsAnalyzer::SportsAnalyzer(const SportsConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SportsAnalyzer::~SportsAnalyzer() = default;
SportsAnalyzer::SportsAnalyzer(SportsAnalyzer&&) noexcept = default;
SportsAnalyzer& SportsAnalyzer::operator=(SportsAnalyzer&&) noexcept = default;

void SportsAnalyzer::reset() {
    if (pimpl_) {
        pimpl_->tracker_->reset();
        pimpl_->track_history_.clear();
        pimpl_->possession_team_a_frames_ = 0;
        pimpl_->possession_team_b_frames_ = 0;
        pimpl_->total_frames_ = 0;
    }
}

GameAnalytics SportsAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SportsAnalyzer is null.");

    GameAnalytics analytics;
    analytics.tactical_view = image.clone(); // For drawing
    analytics.player_in_possession_id = -1;

    // 1. Preprocess
    preproc::ImageFrame frame{image.data, image.cols, image.rows, preproc::ImageFormat::BGR};
    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect
    auto raw_dets = pimpl_->detector_->process({pimpl_->output_tensor});

    // Scale to image size
    float sx = (float)image.cols / pimpl_->config_.input_width;
    float sy = (float)image.rows / pimpl_->config_.input_height;
    for (auto& d : raw_dets) {
        d.x1 *= sx; d.x2 *= sx; d.y1 *= sy; d.y2 *= sy;
    }

    // 4. Track
    auto tracks = pimpl_->tracker_->update(raw_dets);

    // 5. Analysis
    analytics.is_ball_visible = false;
    const postproc::TrackedObject* ball = nullptr;

    for (const auto& t : tracks) {
        Player p;
        p.track_id = t.track_id;
        p.position.x = (t.box.x1 + t.box.x2) / 2.0f;
        p.position.y = t.box.y2; // Bottom of box for ground position

        // Project to field coords
        // p.position = pimpl_->to_field_coords(p.position);

        // Update history
        auto& hist = pimpl_->track_history_[t.track_id];
        hist.class_id = t.box.class_id;
        hist.history.push_back(p.position);
        if (hist.history.size() > 50) hist.history.erase(hist.history.begin());

        p.trajectory = hist.history;

        // Update team ID
        if (t.box.class_id == pimpl_->config_.team_a_class_id) hist.team_id = 0;
        else if (t.box.class_id == pimpl_->config_.team_b_class_id) hist.team_id = 1;

        p.team_id = hist.team_id;

        // Assign to correct team
        if (t.box.class_id == pimpl_->config_.team_a_class_id) analytics.team_a.push_back(p);
        else if (t.box.class_id == pimpl_->config_.team_b_class_id) analytics.team_b.push_back(p);
        else if (t.box.class_id == pimpl_->config_.ball_class_id) {
            analytics.is_ball_visible = true;
            analytics.ball_position = p.position;
            ball = &t;
        }

        // Draw trajectory
        if (p.trajectory.size() > 1) {
            cv::Scalar color = (p.team_id == 0) ? cv::Scalar(255,0,0) : cv::Scalar(0,255,255);
            for(size_t i=1; i<p.trajectory.size(); ++i) {
                cv::line(analytics.tactical_view, p.trajectory[i-1], p.trajectory[i], color, 2);
            }
        }
    }

    // Possession Logic
    if (ball) {
        float min_dist = 1e9;
        int closest_player_id = -1;
        int closest_team = -1;

        auto check_team = [&](const std::vector<Player>& team) {
            for (const auto& p : team) {
                float dist = cv::norm(p.position - analytics.ball_position);
                if (dist < min_dist) {
                    min_dist = dist;
                    closest_player_id = p.track_id;
                    closest_team = p.team_id;
                }
            }
        };

        check_team(analytics.team_a);
        check_team(analytics.team_b);

        // If ball is close to a player (e.g. < 30 pixels)
        if (min_dist < 30.0f) {
            analytics.player_in_possession_id = closest_player_id;
            if (closest_team == 0) pimpl_->possession_team_a_frames_++;
            else pimpl_->possession_team_b_frames_++;
        }
    }

    // Update global possession
    pimpl_->total_frames_++;
    if (pimpl_->total_frames_ > 0) {
        analytics.team_a_possession_percent = (float)pimpl_->possession_team_a_frames_ / pimpl_->total_frames_ * 100.0f;
    }

    return analytics;
}

} // namespace xinfer::zoo::live_events