#include <include/zoo/document/signature_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/detection.h>
#include <include/postproc/yolo_decoder.h>

namespace xinfer::zoo::document {

const int MAX_DECODED_SIGNATURES = 256;

struct SignatureDetector::Impl {
    SignatureDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;

    core::Tensor decoded_boxes_gpu;
    core::Tensor decoded_scores_gpu;
    core::Tensor decoded_classes_gpu;

    std::vector<float> h_boxes;
    std::vector<float> h_scores;

    Impl(const SignatureDetectorConfig& config) : config_(config) {
        decoded_boxes_gpu = core::Tensor({MAX_DECODED_SIGNATURES, 4}, core::DataType::kFLOAT);
        decoded_scores_gpu = core::Tensor({MAX_DECODED_SIGNATURES}, core::DataType::kFLOAT);
        decoded_classes_gpu = core::Tensor({MAX_DECODED_SIGNATURES}, core::DataType::kINT32);
        h_boxes.resize(MAX_DECODED_SIGNATURES * 4);
        h_scores.resize(MAX_DECODED_SIGNATURES);
    }
};

SignatureDetector::SignatureDetector(const SignatureDetectorConfig& config) : pimpl_(new Impl(config)) {
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Signature detector engine file not found: " + pimpl_->config_.engine_path);
    }
    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(pimpl_->config_.input_width, pimpl_->config_.input_height, true);
}

SignatureDetector::~SignatureDetector() = default;
SignatureDetector::SignatureDetector(SignatureDetector&&) noexcept = default;
SignatureDetector& SignatureDetector::operator=(SignatureDetector&&) noexcept = default;

std::vector<Signature> SignatureDetector::predict(const cv::Mat& document_image) {
    if (!pimpl_) throw std::runtime_error("SignatureDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(document_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& raw_output = output_tensors[0];

    postproc::yolo::decode(raw_output, pimpl_->config_.confidence_threshold,
                           pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->decoded_classes_gpu);

    std::vector<int> nms_indices = postproc::detection::nms(
        pimpl_->decoded_boxes_gpu, pimpl_->decoded_scores_gpu, pimpl_->config_.nms_iou_threshold);

    std::vector<Signature> final_signatures;
    if (nms_indices.empty()) {
        return final_signatures;
    }

    pimpl_->decoded_boxes_gpu.copy_to_host(pimpl_->h_boxes.data());
    pimpl_->decoded_scores_gpu.copy_to_host(pimpl_->h_scores.data());

    float scale_x = (float)document_image.cols / pimpl_->config_.input_width;
    float scale_y = (float)document_image.rows / pimpl_->config_.input_height;

    for (int idx : nms_indices) {
        Signature sig;
        float x1 = pimpl_->h_boxes[idx * 4 + 0] * scale_x;
        float y1 = pimpl_->h_boxes[idx * 4 + 1] * scale_y;
        float x2 = pimpl_->h_boxes[idx * 4 + 2] * scale_x;
        float y2 = pimpl_->h_boxes[idx * 4 + 3] * scale_y;
        sig.bounding_box = cv::Rect(cv::Point((int)x1, (int)y1), cv::Point((int)x2, (int)y2));
        sig.confidence = pimpl_->h_scores[idx];

        final_signatures.push_back(sig);
    }

    return final_signatures;
}

} // namespace xinfer::zoo::document```