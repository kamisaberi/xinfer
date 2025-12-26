#include <xinfer/zoo/maritime/collision_avoidance.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <cmath>
#include <map>

namespace xinfer::zoo::maritime {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CollisionAvoidance::Impl {
    CollisionConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // State
    struct TrackHistory {
        float last_distance = -1.0f;
    };
    std::map<int, TrackHistory> track_state_;

    Impl(const CollisionConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CollisionAvoidance: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Detector (YOLO)
        detector_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = 0.5f;
        det_cfg.nms_threshold = 0.45f;
        detector_->init(det_cfg);

        // 4. Setup Tracker
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = 30; // Objects at sea persist longer
        trk_cfg.min_hits = 3;
        tracker_->init(trk_cfg);
    }

    // --- Distance Calculation (Pin-hole Flat Water Model) ---
    float estimate_distance(const postproc::BoundingBox& box, int img_height) {
        // Contact point with water is usually the bottom of the bounding box
        float y_contact = box.y2;

        // Distance from horizon in pixels
        float delta_y = y_contact - config_.horizon_y_px;

        if (delta_y <= 1.0f) return 10000.0f; // Infinite/Far away (on or above horizon)

        // D = (H_cam * f) / delta_y
        float dist = (config_.camera_height_m * config_.focal_length_px) / delta_y;
        return dist;
    }

    // --- Bearing Calculation ---
    float estimate_bearing(const postproc::BoundingBox& box, int img_width) {
        float x_center = (box.x1 + box.x2) / 2.0f;
        float offset_x = x_center - (img_width / 2.0f);

        // theta = atan(offset / f)
        float rad = std::atan2(offset_x, config_.focal_length_px);
        return rad * 180.0f / 3.14159f; // Degrees
    }
};

// =================================================================================
// Public API
// =================================================================================

CollisionAvoidance::CollisionAvoidance(const CollisionConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CollisionAvoidance::~CollisionAvoidance() = default;
CollisionAvoidance::CollisionAvoidance(CollisionAvoidance&&) noexcept = default;
CollisionAvoidance& CollisionAvoidance::operator=(CollisionAvoidance&&) noexcept = default;

void CollisionAvoidance::reset() {
    if (pimpl_) {
        pimpl_->tracker_->reset();
        pimpl_->track_state_.clear();
    }
}

AvoidanceResult CollisionAvoidance::process(const cv::Mat& image, float dt_sec) {
    if (!pimpl_) throw std::runtime_error("CollisionAvoidance is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect
    auto raw_dets = pimpl_->detector_->process({pimpl_->output_tensor});

    // Scale detections to image size
    std::vector<postproc::BoundingBox> scaled_dets;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (auto d : raw_dets) {
        d.x1 *= scale_x; d.x2 *= scale_x;
        d.y1 *= scale_y; d.y2 *= scale_y;
        scaled_dets.push_back(d);
    }

    // 4. Track
    auto tracks = pimpl_->tracker_->update(scaled_dets);

    // 5. Risk Analysis
    AvoidanceResult result;
    result.status = RiskLevel::SAFE;
    result.suggested_rudder_angle = 0.0f;

    float min_ttc = 9999.0f;
    float critical_bearing = 0.0f;

    for (const auto& t : tracks) {
        Obstacle obs;
        obs.track_id = t.track_id;
        obs.box = t.box;
        obs.type = (t.box.class_id == 0) ? "Ship" : "Obstacle"; // Simplified mapping

        // Geometry
        obs.distance_m = pimpl_->estimate_distance(t.box, image.rows);
        obs.bearing_deg = pimpl_->estimate_bearing(t.box, image.cols);

        // Velocity & TTC
        auto& hist = pimpl_->track_state_[t.track_id];

        if (hist.last_distance > 0 && dt_sec > 0) {
            float delta_dist = hist.last_distance - obs.distance_m;
            obs.relative_speed = delta_dist / dt_sec; // Positive if approaching

            if (obs.relative_speed > 0.1f) {
                obs.ttc = obs.distance_m / obs.relative_speed;
            } else {
                obs.ttc = 9999.0f; // Not approaching
            }
        } else {
            obs.relative_speed = 0.0f;
            obs.ttc = 9999.0f;
        }

        // Update History
        hist.last_distance = obs.distance_m;

        // Risk Logic
        if (obs.ttc < pimpl_->config_.critical_ttc_sec || obs.distance_m < 20.0f) {
            result.status = RiskLevel::CRITICAL;
            if (obs.ttc < min_ttc) {
                min_ttc = obs.ttc;
                critical_bearing = obs.bearing_deg;
            }
        } else if (obs.distance_m < pimpl_->config_.warning_distance_m) {
            if (result.status != RiskLevel::CRITICAL) result.status = RiskLevel::WARNING;
        }

        result.obstacles.push_back(obs);
    }

    // 6. Suggest Maneuver (Naive P-Controller)
    if (result.status == RiskLevel::CRITICAL) {
        // If object is to the right (+ve bearing), turn Left (-ve angle).
        // If object is to the left (-ve bearing), turn Right (+ve angle).
        result.suggested_rudder_angle = (critical_bearing > 0) ? -45.0f : 45.0f;
    }

    return result;
}

} // namespace xinfer::zoo::maritime