#include <include/zoo/document/handwriting_recognizer.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/core/tensor.h>
#include <include/postproc/ctc_decoder.h>

namespace xinfer::zoo::document {

struct HandwritingRecognizer::Impl {
    HandwritingRecognizerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::vector<std::string> character_map_;
};

HandwritingRecognizer::HandwritingRecognizer(const HandwritingRecognizerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Handwriting recognizer engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (!pimpl_->config_.character_map_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.character_map_path);
        if (!labels_file) throw std::runtime_error("Could not open character map file: " + pimpl_->config_.character_map_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->character_map_.push_back(line);
        }
    }
}

HandwritingRecognizer::~HandwritingRecognizer() = default;
HandwritingRecognizer::HandwritingRecognizer(HandwritingRecognizer&&) noexcept = default;
HandwritingRecognizer& HandwritingRecognizer::operator=(HandwritingRecognizer&&) noexcept = default;

HandwritingRecognitionResult HandwritingRecognizer::predict(const cv::Mat& line_image) {
    if (!pimpl_) throw std::runtime_error("HandwritingRecognizer is in a moved-from state.");

    cv::Mat gray_image;
    if (line_image.channels() == 3) {
        cv::cvtColor(line_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = line_image;
    }

    int target_h = pimpl_->config_.input_height;
    int target_w = static_cast<int>((float)target_h / gray_image.rows * gray_image.cols);

    cv::Mat resized_image;
    cv::resize(gray_image, resized_image, cv::Size(target_w, target_h));

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255.0);

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    input_shape[0] = 1;
    input_shape[2] = target_h;
    input_shape[3] = target_w;

    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    input_tensor.copy_from_host(resized_image.data);

    pimpl_->engine_->setInputShape("input", input_shape);
    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logits_tensor = output_tensors[0];

    auto decoded_result = postproc::ctc::decode(logits_tensor, pimpl_->character_map_);

    HandwritingRecognitionResult result;
    result.text = decoded_result.first;
    result.confidence = decoded_result.second;

    return result;
}

} // namespace xinfer::zoo::document