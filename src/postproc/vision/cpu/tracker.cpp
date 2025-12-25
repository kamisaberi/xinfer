#include "tracker.h"
#include <xinfer/core/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <algorithm>
#include <cmath>
#include <map>

namespace xinfer::postproc {

// =================================================================================
// Internal Helper: KalmanTrack
// =================================================================================

class KalmanTrack {
public:
    cv::KalmanFilter kf;
    int track_id;
    int age = 0;           // Frames since first appearance
    int time_since_update = 0; // Frames since last detection match
    int hits = 0;          // Total successful matches
    BoundingBox last_box;

    KalmanTrack(const BoundingBox& box, int id) : track_id(id) {
        // State: [u, v, s, r, u', v', s']
        // u, v: center x, y
        // s: area (scale)
        // r: aspect ratio (w/h)
        int stateNum = 7;
        int measureNum = 4;

        kf.init(stateNum, measureNum, 0);

        // Transition Matrix (F)
        kf.transitionMatrix = cv::Mat::eye(stateNum, stateNum, CV_32F);
        // Velocity influence
        kf.transitionMatrix.at<float>(0, 4) = 1.0f;
        kf.transitionMatrix.at<float>(1, 5) = 1.0f;
        kf.transitionMatrix.at<float>(2, 6) = 1.0f;

        // Measurement Matrix (H)
        kf.measurementMatrix = cv::Mat::eye(measureNum, stateNum, CV_32F);

        // Process Noise Covariance (Q)
        cv::setIdentity(kf.processNoiseCov, cv::Scalar::all(1e-2));
        kf.processNoiseCov.at<float>(2, 2) *= 10.0f; // Scale changes harder
        kf.processNoiseCov.at<float>(6, 6) *= 10.0f;

        // Measurement Noise Covariance (R)
        cv::setIdentity(kf.measurementNoiseCov, cv::Scalar::all(1e-1));
        kf.measurementNoiseCov.at<float>(2, 2) *= 10.0f;

        // Error Covariance (P)
        cv::setIdentity(kf.errorCovPost, cv::Scalar::all(1.0f));
        kf.errorCovPost.at<float>(2, 2) *= 10.0f;
        kf.errorCovPost.at<float>(6, 6) *= 10.0f;

        // Initialize state
        // Convert [x1, y1, x2, y2] -> [cx, cy, s, r]
        float w = box.x2 - box.x1;
        float h = box.y2 - box.y1;
        float cx = box.x1 + w / 2.0f;
        float cy = box.y1 + h / 2.0f;
        float s = w * h;
        float r = w / (h + 1e-6f); // Avoid div/0

        kf.statePost.at<float>(0) = cx;
        kf.statePost.at<float>(1) = cy;
        kf.statePost.at<float>(2) = s;
        kf.statePost.at<float>(3) = r;

        last_box = box;
    }

    BoundingBox predict() {
        cv::Mat p = kf.predict();

        // Convert State -> Box
        float cx = p.at<float>(0);
        float cy = p.at<float>(1);
        float s = p.at<float>(2);
        float r = p.at<float>(3);

        float w = std::sqrt(s * r);
        float h = s / (w + 1e-6f); // s = w*h -> h = s/w

        BoundingBox pred_box;
        pred_box.x1 = cx - w / 2.0f;
        pred_box.y1 = cy - h / 2.0f;
        pred_box.x2 = cx + w / 2.0f;
        pred_box.y2 = cy + h / 2.0f;
        pred_box.class_id = last_box.class_id;
        pred_box.confidence = last_box.confidence; // Decay logic could be added here

        return pred_box;
    }

    void update(const BoundingBox& det) {
        time_since_update = 0;
        hits++;
        last_box = det;

        float w = det.x2 - det.x1;
        float h = det.y2 - det.y1;
        float cx = det.x1 + w / 2.0f;
        float cy = det.y1 + h / 2.0f;
        float s = w * h;
        float r = w / (h + 1e-6f);

        cv::Mat measurement(4, 1, CV_32F);
        measurement.at<float>(0) = cx;
        measurement.at<float>(1) = cy;
        measurement.at<float>(2) = s;
        measurement.at<float>(3) = r;

        kf.correct(measurement);
    }

    // Get estimated velocity for Aegis Sky leading shots
    void get_velocity(float& vx, float& vy) const {
        vx = kf.statePost.at<float>(4);
        vy = kf.statePost.at<float>(5);
    }
};

// =================================================================================
// IoU Helper
// =================================================================================

static float calculate_iou(const BoundingBox& a, const BoundingBox& b) {
    float xx1 = std::max(a.x1, b.x1);
    float yy1 = std::max(a.y1, b.y1);
    float xx2 = std::min(a.x2, b.x2);
    float yy2 = std::min(a.y2, b.y2);

    float w = std::max(0.0f, xx2 - xx1);
    float h = std::max(0.0f, yy2 - yy1);
    float inter = w * h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    return inter / (area_a + area_b - inter + 1e-6f);
}

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CpuTracker::Impl {
    TrackerConfig config;
    std::vector<KalmanTrack> tracks;
    int next_id = 1;

    void reset() {
        tracks.clear();
        next_id = 1;
    }
};

// =================================================================================
// Class Implementation
// =================================================================================

CpuTracker::CpuTracker() : m_impl(std::make_unique<Impl>()) {}
CpuTracker::~CpuTracker() = default;

void CpuTracker::init(const TrackerConfig& config) {
    m_impl->config = config;
    reset();
}

void CpuTracker::reset() {
    m_impl->reset();
}

std::vector<TrackedObject> CpuTracker::update(const std::vector<BoundingBox>& detections) {
    auto& tracks = m_impl->tracks;

    // 1. Predict Phase
    std::vector<BoundingBox> predictions;
    for (auto& t : tracks) {
        t.age++;
        t.time_since_update++;
        predictions.push_back(t.predict());
    }

    // 2. Association Phase (Greedy Strategy)
    // We try to match detections to predictions based on IoU.
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(tracks.size(), false);

    // Sort matching pairs by IoU desc to prevent bad greedy matches?
    // For simplicity, simple O(N*M) loop here.

    // We iterate tracks first to find best detection for each track
    for (size_t t = 0; t < tracks.size(); ++t) {
        float best_iou = 0.0f;
        int best_det_idx = -1;

        for (size_t d = 0; d < detections.size(); ++d) {
            if (det_matched[d]) continue; // Already matched

            // Optimization: Only match same class ID
            if (detections[d].class_id != tracks[t].last_box.class_id) continue;

            float iou = calculate_iou(predictions[t], detections[d]);
            if (iou > best_iou) {
                best_iou = iou;
                best_det_idx = d;
            }
        }

        if (best_iou >= m_impl->config.iou_threshold && best_det_idx != -1) {
            tracks[t].update(detections[best_det_idx]);
            track_matched[t] = true;
            det_matched[best_det_idx] = true;
        }
    }

    // 3. New Tracks
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!det_matched[d]) {
            // Only start new track if confidence is reasonably high to avoid noise
            if (detections[d].confidence > 0.4f) {
                m_impl->tracks.emplace_back(detections[d], m_impl->next_id++);
            }
        }
    }

    // 4. Cleanup & Output
    std::vector<TrackedObject> active_objects;
    auto it = m_impl->tracks.begin();

    while (it != m_impl->tracks.end()) {
        bool remove = false;

        // Remove if too old without update
        if (it->time_since_update > m_impl->config.max_age) {
            remove = true;
        }

        // Remove 'tentative' tracks that age out before getting enough hits
        // (Prevents flickering noise from becoming a track)
        if (it->time_since_update > 0 && it->hits < m_impl->config.min_hits) {
            // If it's young and missed a frame, kill it
            // remove = true; // Strict logic
        }

        if (remove) {
            it = m_impl->tracks.erase(it);
        } else {
            // Only output "confirmed" tracks
            if (it->hits >= m_impl->config.min_hits || it->time_since_update == 0) {
                TrackedObject obj;
                obj.track_id = it->track_id;
                obj.box = it->last_box; // Use actual detection if matched, else use prediction?

                // If we matched this frame, use detection (accurate).
                // If we missed (coasting), use prediction.
                if (it->time_since_update > 0) {
                    obj.box = predictions[std::distance(m_impl->tracks.begin(), it)];
                }

                obj.age = it->age;
                obj.hits = it->hits;

                // Get Velocity for Lead Calculation
                it->get_velocity(obj.velocity_x, obj.velocity_y);

                active_objects.push_back(obj);
            }
            ++it;
        }
    }

    return active_objects;
}

} // namespace xinfer::postproc