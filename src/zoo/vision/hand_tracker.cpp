#include <include/zoo/vision/hand_tracker.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>

namespace xinfer::zoo::vision {

const int MAX_DECODED_HANDS = 1024;

struct TrackedHand {
    int id;
    float x1, y1, x2, y2;
    float confidence;
    std::vector<cv::Point2f> keypoints;
    int age = 0;
};

float calculate_hand_iou(const TrackedHand& track, const std::vector<float>& box) {
    float ix1 = std::max(track.x1, box[0]);
    float iy1 = std::max(track.y1, box[1]);
    float ix2 = std::min(track.x2, box[2]);
    float iy2 = std::min(track.y2, box[3]);
    float inter_area = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    float track_area = (track.x2 - track.x1) * (track.y2 - track.y1);
    float box_area = (box[2] - box[0]) * (box[3] - box[1]);
    float union_area = track_area + box_area - inter_area;
    return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
}

struct HandTracker::Impl {
    HandTrackerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;

    std::vector<TrackedHand> active_tracks;
    int next_track_id = 0;
};

HandTracker::HandTracker(const HandTrackerConfig& config) : pimpl_(new Impl{config}) {
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("Hand detection engine file not found: " + pimpl_->config_.detection_engine_path);
    }
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(pimpl_->config_.input_width, pimpl_->config_.input_height, true);
}

HandTracker::~HandTracker() = default;
HandTracker::HandTracker(HandTracker&&) noexcept = default;
HandTracker& HandTracker::operator=(HandTracker&&) noexcept = default;

std::vector<Hand> HandTracker::track(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("HandTracker is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // Assume the model is a detector that gives boxes and keypoints
    // A full implementation would use a dedicated decoder and NMS here.
    // This is a simplified placeholder for the post-processing logic.
    std::vector<std::vector<float>> current_detections; // This would be populated from model output

    std::vector<bool> matched_detections(current_detections.size(), false);
    std::vector<bool> matched_tracks(pimpl_->active_tracks.size(), false);

    for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
        float best_iou = 0.0f;
        int best_match_idx = -1;
        for (size_t j = 0; j < current_detections.size(); ++j) {
            if (matched_detections[j]) continue;
            float iou = calculate_hand_iou(pimpl_->active_tracks[i], current_detections[j]);
            if (iou > best_iou) {
                best_iou = iou;
                best_match_idx = j;
            }
        }
        if (best_iou > 0.4f) {
            pimpl_->active_tracks[i].x1 = current_detections[best_match_idx][0];
            pimpl_->active_tracks[i].y1 = current_detections[best_match_idx][1];
            pimpl_->active_tracks[i].x2 = current_detections[best_match_idx][2];
            pimpl_->active_tracks[i].y2 = current_detections[best_match_idx][3];
            pimpl_->active_tracks[i].confidence = current_detections[best_match_idx][4];
            pimpl_->active_tracks[i].age = 0;
            // Keypoint update logic would go here
            matched_detections[best_match_idx] = true;
            matched_tracks[i] = true;
        }
    }

    for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
        if (!matched_tracks[i]) pimpl_->active_tracks[i].age++;
    }

    for (size_t j = 0; j < current_detections.size(); ++j) {
        if (!matched_detections[j]) {
            TrackedHand new_track;
            new_track.id = pimpl_->next_track_id++;
            new_track.x1 = current_detections[j][0];
            new_track.y1 = current_detections[j][1];
            new_track.x2 = current_detections[j][2];
            new_track.y2 = current_detections[j][3];
            new_track.confidence = current_detections[j][4];
            pimpl_->active_tracks.push_back(new_track);
        }
    }

    pimpl_->active_tracks.erase(std::remove_if(pimpl_->active_tracks.begin(), pimpl_->active_tracks.end(),
        [](const TrackedHand& t) { return t.age > 20; }), pimpl_->active_tracks.end());

    std::vector<Hand> results;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& track : pimpl_->active_tracks) {
        Hand res;
        res.track_id = track.id;
        res.confidence = track.confidence;
        res.x1 = track.x1 * scale_x;
        res.y1 = track.y1 * scale_y;
        res.x2 = track.x2 * scale_x;
        res.y2 = track.y2 * scale_y;
        for(const auto& pt : track.keypoints){
             res.keypoints.push_back({pt.x * scale_x, pt.y * scale_y});
        }
        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::vision