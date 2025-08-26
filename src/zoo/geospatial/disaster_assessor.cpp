#include <include/zoo/geospatial/disaster_assessor.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::geospatial {

struct DisasterAssessor::Impl {
    DisasterAssessorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

DisasterAssessor::DisasterAssessor(const DisasterAssessorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Disaster assessor engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Disaster assessor engine must have exactly two inputs.");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        pimpl_->config_.mean,
        pimpl_->config_.std
    );
}

DisasterAssessor::~DisasterAssessor() = default;
DisasterAssessor::DisasterAssessor(DisasterAssessor&&) noexcept = default;
DisasterAssessor& DisasterAssessor::operator=(DisasterAssessor&&) noexcept = default;

cv::Mat DisasterAssessor::predict(const cv::Mat& image_before, const cv::Mat& image_after) {
    if (!pimpl_) throw std::runtime_error("DisasterAssessor is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor_before(input_shape, core::DataType::kFLOAT);
    core::Tensor input_tensor_after(input_shape, core::DataType::kFLOAT);

    pimpl_->preprocessor_->process(image_before, input_tensor_before);
    pimpl_->preprocessor_->process(image_after, input_tensor_after);

    auto output_tensors = pimpl_->engine_->infer({input_tensor_before, input_tensor_after});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    auto output_shape = logit_map_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> logits(logit_map_tensor.num_elements());
    logit_map_tensor.copy_to_host(logits.data());

    cv::Mat probability_map(H, W, CV_32F, logits.data());

    cv::exp(-probability_map, probability_map);
    probability_map = 1.0 / (1.0 + probability_map);

    cv::Mat binary_mask;
    cv::threshold(probability_map, binary_mask, pimpl_->config_.damage_threshold, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8UC1);

    cv::Mat final_mask;
    cv::resize(binary_mask, final_mask, image_before.size(), 0, 0, cv::INTER_NEAREST);

    return final_mask;
}

} // namespace xinfer::zoo::geospatial