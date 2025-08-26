#include <include/zoo/geospatial/maritime_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::geospatial {

const int MAX_DECODED_MARITIME_OBJECTS = 2048;

struct MaritimeDetector::Impl {
    MaritimeDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;
    std::vector<int> h_classes;

    Impl(const MaritimeDetectorConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_MARITIME_OBJECTS, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_MARITIME_OBJECTS}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_MARITIME_OBJECTS}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_MARITIME_OBJECTS * 4);
        h_scores.resize(MAX_DECODED_MARITIME_OBJECTS);
        h_classes.resize(MAX_DECODED_MARITIME_OBJECTS);
    }
};

MaritimeDetector::MaritimeDetector(const MaritimeDetectorConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Maritime detector engine file not found: " + pimpl_->config_.engine_path);
    }
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
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

MaritimeDetector::~MaritimeDetector() = default;
MaritimeDetector::MaritimeDetector(MaritimeDetector&&) noexcept = default;
MaritimeDetector& MaritimeDetector::operator=(MaritimeDetector&&) noexcept = default;

std::vector<DetectedObject> MaritimeDetector::predict(const cv::Mat& satellite_image) {
    if (!pimpl_) throw std::runtime_error("MaritimeDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(satellite_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    std::vector<DetectedObject> final_objects;
    if (nms_indices.empty()) {
        return final_objects;
    }

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());
    pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes.data());

    float scale_x = (float)satellite_image.cols / pimpl_->config_.input_width;
    float scale_y = (float)satellite_image.rows / pimpl_->config_.input_height;

    for (int idx : nms_indices) {
        DetectedObject obj;
        float x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        float y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        float x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        float y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;

        obj.confidence = pimpl_->h_scores[idx];
        obj.class_id = pimpl_->h_classes[idx];

        // For rotated boxes, a different decoder would be needed.
        // This is a simplified conversion to a contour.
        obj.contour = {{x1, y1}, {x2, y1}, {x2, y2}, {x1, y2}};

        if (!pimpl_->class_labels_.empty() && obj.class_id < pimpl_->class_labels_.size()) {
            obj.label = pimpl_->class_labels_[obj.class_id];
        }

        final_objects.push_back(obj);
    }

    return final_objects;
}

} // namespace xinfer::zoo::geospatial