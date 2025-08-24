#include <include/zoo/vision/animal_tracker.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <map>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::vision {

const int MAX_DECODED_BOXES_TRACKER = 4096;
const int MAX_TRACK_AGE = 30; // Frames

struct Track {
    int id;
    float x1, y1, x2, y2;
    int class_id;
    float confidence;
    int age = 0;
};

float calculate_iou(const Track& track, const std::vector<float>& box) {
    float intersection_x1 = std::max(track.x1, box[0]);
    float intersection_y1 = std::max(track.y1, box[1]);
    float intersection_x2 = std::min(track.x2, box[2]);
    float intersection_y2 = std::min(track.y2, box[3]);

    float intersection_area = std::max(0.0f, intersection_x2 - intersection_x1) * std::max(0.0f, intersection_y2 - intersection_y1);
    float track_area = (track.x2 - track.x1) * (track.y2 - track.y1);
    float box_area = (box[2] - box[0]) * (box[3] - box[1]);

    float union_area = track_area + box_area - intersection_area;
    return (union_area > 0.0f) ? intersection_area / union_area : 0.0f;
}

struct AnimalTracker::Impl {
    AnimalTrackerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;
    std::vector<int> h_classes;

    std::vector<Track> active_tracks;
    int next_track_id = 0;

    Impl(const AnimalTrackerConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_BOXES_TRACKER, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_BOXES_TRACKER}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_BOXES_TRACKER}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_BOXES_TRACKER * 4);
        h_scores.resize(MAX_DECODED_BOXES_TRACKER);
        h_classes.resize(MAX_DECODED_BOXES_TRACKER);
    }
};

AnimalTracker::AnimalTracker(const AnimalTrackerConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("Detection engine file not found: " + pimpl_->config_.detection_engine_path);
    }
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(pimpl_->config_.input_width, pimpl_->config_.input_height, true);
    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

AnimalTracker::~AnimalTracker() = default;
AnimalTracker::AnimalTracker(AnimalTracker&&) noexcept = default;
AnimalTracker& AnimalTracker::operator=(AnimalTracker&&) noexcept = default;

std::vector<TrackedAnimal> AnimalTracker::track(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("AnimalTracker is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    if (nms_indices.empty()) {
        for (auto& track : pimpl_->active_tracks) {
            track.age++;
        }
    } else {
        pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
        pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());
        pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes.data());

        std::vector<bool> matched_detections(nms_indices.size(), false);
        std::vector<bool> matched_tracks(pimpl_->active_tracks.size(), false);

        for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
            float best_iou = 0.0f;
            int best_match_idx = -1;
            for (size_t j = 0; j < nms_indices.size(); ++j) {
                if (matched_detections[j]) continue;

                int det_idx = nms_indices[j];
                std::vector<float> det_box = {pimpl_->h_boxes[det_idx * 4], pimpl_->h_boxes[det_idx * 4 + 1], pimpl_->h_boxes[det_idx * 4 + 2], pimpl_->h_boxes[det_idx * 4 + 3]};
                float iou = calculate_iou(pimpl_->active_tracks[i], det_box);
                if (iou > best_iou) {
                    best_iou = iou;
                    best_match_idx = j;
                }
            }

            if (best_iou > 0.3f) {
                int det_idx = nms_indices[best_match_idx];
                pimpl_->active_tracks[i].x1 = pimpl_->h_boxes[det_idx * 4 + 0];
                pimpl_->active_tracks[i].y1 = pimpl_->h_boxes[det_idx * 4 + 1];
                pimpl_->active_tracks[i].x2 = pimpl_->h_boxes[det_idx * 4 + 2];
                pimpl_->active_tracks[i].y2 = pimpl_->h_boxes[det_idx * 4 + 3];
                pimpl_->active_tracks[i].confidence = pimpl_->h_scores[det_idx];
                pimpl_->active_tracks[i].class_id = pimpl_->h_classes[det_idx];
                pimpl_->active_tracks[i].age = 0;
                matched_detections[best_match_idx] = true;
                matched_tracks[i] = true;
            }
        }

        for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
            if (!matched_tracks[i]) {
                pimpl_->active_tracks[i].age++;
            }
        }

        for (size_t j = 0; j < nms_indices.size(); ++j) {
            if (!matched_detections[j]) {
                int det_idx = nms_indices[j];
                Track new_track;
                new_track.id = pimpl_->next_track_id++;
                new_track.x1 = pimpl_->h_boxes[det_idx * 4 + 0];
                new_track.y1 = pimpl_->h_boxes[det_idx * 4 + 1];
                new_track.x2 = pimpl_->h_boxes[det_idx * 4 + 2];
                new_track.y2 = pimpl_->h_boxes[det_idx * 4 + 3];
                new_track.confidence = pimpl_->h_scores[det_idx];
                new_track.class_id = pimpl_->h_classes[det_idx];
                pimpl_->active_tracks.push_back(new_track);
            }
        }
    }

    pimpl_->active_tracks.erase(std::remove_if(pimpl_->active_tracks.begin(), pimpl_->active_tracks.end(),
        [](const Track& t) { return t.age > MAX_TRACK_AGE; }), pimpl_->active_tracks.end());

    std::vector<TrackedAnimal> results;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& track : pimpl_->active_tracks) {
        TrackedAnimal res;
        res.track_id = track.id;
        res.class_id = track.class_id;
        res.confidence = track.confidence;
        res.x1 = track.x1 * scale_x;
        res.y1 = track.y1 * scale_y;
        res.x2 = track.x2 * scale_x;
        res.y2 = track.y2 * scale_y;
        if (res.class_id < pimpl_->class_labels_.size()) {
            res.label = pimpl_->class_labels_[res.class_id];
        }
        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::vision