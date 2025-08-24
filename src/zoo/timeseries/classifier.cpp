#include <include/zoo/timeseries/classifier.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>
#include <include/core/tensor.h>

namespace xinfer::zoo::timeseries {

struct Classifier::Impl {
    ClassifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::vector<std::string> class_labels_;
};

Classifier::Classifier(const ClassifierConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Time-series classifier engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

Classifier::~Classifier() = default;
Classifier::Classifier(Classifier&&) noexcept = default;
Classifier& Classifier::operator=(Classifier&&) noexcept = default;

TimeSeriesClassificationResult Classifier::predict(const std::vector<float>& time_series_window) {
    if (!pimpl_) throw std::runtime_error("Classifier is in a moved-from state.");
    if (time_series_window.size() != pimpl_->config_.sequence_length) {
        throw std::invalid_argument("Input time_series_window size does not match model's expected sequence_length.");
    }

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    input_tensor.copy_from_host(time_series_window.data());

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logits_tensor = output_tensors[0];

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

    TimeSeriesClassificationResult result;
    result.class_id = max_idx;
    result.confidence = confidence;
    if (max_idx < pimpl_->class_labels_.size()) {
        result.label = pimpl_->class_labels_[max_idx];
    } else {
        result.label = "Class " + std::to_string(max_idx);
    }

    return result;
}

} // namespace xinfer::zoo::timeseries