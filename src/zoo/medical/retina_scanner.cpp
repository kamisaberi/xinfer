#include <include/zoo/medical/retina_scanner.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::medical {

struct RetinaScanner::Impl {
    RetinaScannerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
    std::vector<std::string> diagnosis_labels_ = {"No DR", "Mild", "Moderate", "Severe", "Proliferative DR"};
};

RetinaScanner::RetinaScanner(const RetinaScannerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Retina scanner engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

RetinaScanner::~RetinaScanner() = default;
RetinaScanner::RetinaScanner(RetinaScanner&&) noexcept = default;
RetinaScanner& RetinaScanner::operator=(RetinaScanner&&) noexcept = default;

RetinaScanResult RetinaScanner::predict(const cv::Mat& fundus_image) {
    if (!pimpl_) throw std::runtime_error("RetinaScanner is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(fundus_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    const core::Tensor& logits_tensor = output_tensors[0];
    const core::Tensor& heatmap_tensor = output_tensors[1];

    std::vector<float> logits(logits_tensor.num_elements());
    logits_tensor.copy_to_host(logits.data());

    auto max_it = std::max_element(logits.begin(), logits.end());
    int max_idx = std::distance(logits.begin(), max_it);
    float max_val = *max_it;

    float sum_exp = 0.0f;
    for (float logit : logits) {
        sum_exp += expf(logit - max_val);
    }
    float confidence = 1.0f / sum_exp;

    RetinaScanResult result;
    result.severity_grade = max_idx;
    result.confidence = confidence;
    if (max_idx < pimpl_->diagnosis_labels_.size()) {
        result.diagnosis = pimpl_->diagnosis_labels_[max_idx];
    }

    auto heatmap_shape = heatmap_tensor.shape();
    cv::Mat heatmap(heatmap_shape[2], heatmap_shape[3], CV_32F);
    heatmap_tensor.copy_to_host(heatmap.data);

    cv::resize(heatmap, result.lesion_heatmap, fundus_image.size(), 0, 0, cv::INTER_LINEAR);

    return result;
}

} // namespace xinfer::zoo::medical