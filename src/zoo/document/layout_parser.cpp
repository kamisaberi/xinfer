#include <include/zoo/document/layout_parser.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::document {

const int MAX_DECODED_LAYOUT_ELEMENTS = 256;

struct LayoutParser::Impl {
    LayoutParserConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> class_labels_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;
    std::vector<int> h_classes;

    Impl(const LayoutParserConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_LAYOUT_ELEMENTS, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_LAYOUT_ELEMENTS}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_LAYOUT_ELEMENTS}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_LAYOUT_ELEMENTS * 4);
        h_scores.resize(MAX_DECODED_LAYOUT_ELEMENTS);
        h_classes.resize(MAX_DECODED_LAYOUT_ELEMENTS);
    }
};

LayoutParser::LayoutParser(const LayoutParserConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Layout parser engine file not found: " + pimpl_->config_.engine_path);
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

LayoutParser::~LayoutParser() = default;
LayoutParser::LayoutParser(LayoutParser&&) noexcept = default;
LayoutParser& LayoutParser::operator=(LayoutParser&&) noexcept = default;

std::vector<LayoutElement> LayoutParser::predict(const cv::Mat& document_image) {
    if (!pimpl_) throw std::runtime_error("LayoutParser is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(document_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    std::vector<LayoutElement> final_elements;
    if (nms_indices.empty()) {
        return final_elements;
    }

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());
    pimpl_->decoded_classes_gpu.copy_to_host(pimpl_->h_classes.data());

    float scale_x = (float)document_image.cols / pimpl_->config_.input_width;
    float scale_y = (float)document_image.rows / pimpl_->config_.input_height;

    for (int idx : nms_indices) {
        LayoutElement element;
        float x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        float y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        float x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        float y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        element.bounding_box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        element.confidence = pimpl_->h_scores[idx];
        element.class_id = pimpl_->h_classes[idx];

        if (!pimpl_->class_labels_.empty() && element.class_id < pimpl_->class_labels_.size()) {
            element.label = pimpl_->class_labels_[element.class_id];
        } else {
            element.label = "Element " + std::to_string(element.class_id);
        }

        final_elements.push_back(element);
    }

    return final_elements;
}

} // namespace xinfer::zoo::document