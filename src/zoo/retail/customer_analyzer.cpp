#include <include/zoo/retail/customer_analyzer.h>
#include <stdexcept>
#include <vector>
#include <map>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::retail {

const int MAX_DECODED_CUSTOMERS = 512;

struct CustomerTrack {
    int id;
    cv::Rect box;
    vision::Pose pose;
    int age = 0;
};

float calculate_customer_iou(const cv::Rect& box1, const cv::Rect& box2) {
    cv::Rect intersection = box1 & box2;
    float inter_area = intersection.area();
    float union_area = box1.area() + box2.area() - inter_area;
    return (union_area > 0.0f) ? inter_area / union_area : 0.0f;
}

struct CustomerAnalyzer::Impl {
    CustomerAnalyzerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_detector_;
    std::unique_ptr<vision::PoseEstimator> pose_estimator_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_detector_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;

    std::vector<CustomerTrack> active_tracks;
    int next_track_id = 0;
    cv::Mat heatmap;
};

CustomerAnalyzer::CustomerAnalyzer(const CustomerAnalyzerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.detection_engine_path).good()) {
        throw std::runtime_error("Customer detector engine file not found: " + pimpl_->config_.detection_engine_path);
    }
    pimpl_->engine_detector_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.detection_engine_path);

    vision::PoseEstimatorConfig pose_config;
    pose_config.engine_path = pimpl_->config_.pose_engine_path;
    pimpl_->pose_estimator_ = std::make_unique<vision::PoseEstimator>(pose_config);

    pimpl_->preprocessor_detector_ = std::make_unique<preproc::ImageProcessor>(pimpl_->config_.detection_input_width, pimpl_->config_.detection_input_height, true);

    pimpl_->decoded_boxes_gpu = core::Tensor({MAX_DECODED_CUSTOMERS, 4}, core::DataType::kFLOAT);
    pimpl_->decoded_scores_gpu = core::Tensor({MAX_DECODED_CUSTOMERS}, core::DataType::kFLOAT);
    pimpl_->decoded_classes_gpu = core::Tensor({MAX_DECODED_CUSTOMERS}, core::DataType::kINT32);
    pimpl_->h_boxes.resize(MAX_DECODED_CUSTOMERS * 4);
    pimpl_->h_scores.resize(MAX_DECODED_CUSTOMERS);
}

CustomerAnalyzer::~CustomerAnalyzer() = default;
CustomerAnalyzer::CustomerAnalyzer(CustomerAnalyzer&&) noexcept = default;
CustomerAnalyzer& CustomerAnalyzer::operator=(CustomerAnalyzer&&) noexcept = default;

std::vector<TrackedCustomer> CustomerAnalyzer::track(const cv::Mat& frame) {
    if (!pimpl_) throw std::runtime_error("CustomerAnalyzer is in a moved-from state.");

    if (pimpl_->heatmap.empty()) {
        pimpl_->heatmap = cv::Mat::zeros(frame.size(), CV_32F);
    }

    auto input_shape = pimpl_->engine_detector_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_detector_->process(frame, input_tensor);

    auto output_tensors = pimpl_->engine_detector_->infer({input_tensor});

    postproc::yolo::decode(output_tensors[0], pimpl_->config_.detection_confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.detection_nms_iou_threshold);

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());

    std::vector<cv::Rect> current_detections;
    float scale_x = (float)frame.cols / pimpl_->config_.detection_input_width;
    float scale_y = (float)frame.rows / pimpl_->config_.detection_input_height;

    for (int idx : nms_indices) {
        float x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        float y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        float x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        float y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        current_detections.emplace_back((int)x1, (int)y1, (int)(x2-x1), (int)(y2-y1));
    }

    std::vector<bool> matched_detections(current_detections.size(), false);
    std::vector<bool> matched_tracks(pimpl_->active_tracks.size(), false);

    for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
        float best_iou = 0.0f;
        int best_match_idx = -1;
        for (size_t j = 0; j < current_detections.size(); ++j) {
            if (matched_detections[j]) continue;
            float iou = calculate_customer_iou(pimpl_->active_tracks[i].box, current_detections[j]);
            if (iou > best_iou) { best_iou = iou; best_match_idx = j; }
        }
        if (best_iou > 0.3f) {
            pimpl_->active_tracks[i].box = current_detections[best_match_idx];
            pimpl_->active_tracks[i].age = 0;
            matched_detections[best_match_idx] = true;
            matched_tracks[i] = true;
        }
    }

    for (size_t i = 0; i < pimpl_->active_tracks.size(); ++i) {
        if (!matched_tracks[i]) pimpl_->active_tracks[i].age++;
    }

    for (size_t j = 0; j < current_detections.size(); ++j) {
        if (!matched_detections[j]) {
            CustomerTrack new_track;
            new_track.id = pimpl_->next_track_id++;
            new_track.box = current_detections[j];
            pimpl_->active_tracks.push_back(new_track);
        }
    }

    pimpl_->active_tracks.erase(std::remove_if(pimpl_->active_tracks.begin(), pimpl_->active_tracks.end(),
        [](const CustomerTrack& t) { return t.age > 30; }), pimpl_->active_tracks.end());

    std::vector<TrackedCustomer> results;
    for (auto& track : pimpl_->active_tracks) {
        cv::Rect roi = track.box & cv::Rect(0, 0, frame.cols, frame.rows);
        if (roi.width > 0 && roi.height > 0) {
            cv::Mat customer_patch = frame(roi);
            track.pose = pimpl_->pose_estimator_->predict(customer_patch)[0];

            TrackedCustomer res;
            res.track_id = track.id;
            res.bounding_box = track.box;
            res.pose = track.pose;
            results.push_back(res);

            cv::Point center = (track.box.tl() + track.box.br()) * 0.5;
            pimpl_->heatmap.at<float>(center) += 1.0f;
        }
    }

    return results;
}

cv::Mat CustomerAnalyzer::generate_heatmap() {
    if (!pimpl_ || pimpl_->heatmap.empty()) return cv::Mat();

    cv::Mat normalized_heatmap;
    cv::normalize(pimpl_->heatmap, normalized_heatmap, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat color_heatmap;
    cv::applyColorMap(normalized_heatmap, color_heatmap, cv::COLORMAP_JET);

    return color_heatmap;
}

} // namespace xinfer::zoo::retail