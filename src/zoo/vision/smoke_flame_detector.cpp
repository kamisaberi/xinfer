#include <include/zoo/vision/smoke_flame_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::vision {

const int MAX_DECODED_EVENTS = 1024;

struct SmokeFlameDetector::Impl {
    SmokeFlameDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;
    std::vector<int> h_classes;

    Impl(const SmokeFlameDetectorConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_EVENTS, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_EVENTS}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_EVENTS}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_EVENTS * 4);
        h_scores.resize(MAX_DECODED_EVENTS);
        h_classes.resize(MAX_DECODED_EVENTS);
    }
};

SmokeFlameDetector::SmokeFlameDetector(const SmokeFlameDetectorConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Smoke/flame detection engine file not found: " + pimpl_->config_.engine_path);
    }
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(pimpl_->config_.input_width, pimpl_->config_.input_height, true);
}

SmokeFlameDetector::~SmokeFlameDetector() = default;
SmokeFlameDetector::SmokeFlameDetector(SmokeFlameDetector&&) noexcept = default;
SmokeFlameDetector& SmokeFlameDetector::operator=(SmokeFlameDetector&&) noexcept = default;

std::vector<DetectionEvent> SmokeFlameDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SmokeFlameDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    std::vector<DetectionEvent> final_events;
    if (nms_indices.empty()) {
        return final_events;
    }

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());
    pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes.data());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (int idx : nms_indices) {
        DetectionEvent event;
        float x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        float y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        float x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        float y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        event.bounding_box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        event.confidence = pimpl_->h_scores[idx];

        int class_id = pimpl_->h_classes[idx];
        if (class_id == 0) { // Assuming class 0 is "smoke"
            event.type = DetectionEvent::Type::SMOKE;
        } else if (class_id == 1) { // Assuming class 1 is "flame"
            event.type = DetectionEvent::Type::FLAME;
        } else {
            continue;
        }

        final_events.push_back(event);
    }

    return final_events;
}

} // namespace xinfer::zoo::vision