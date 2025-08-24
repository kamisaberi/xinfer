#include <include/zoo/audio/classifier.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>

namespace xinfer::zoo::audio {

struct Classifier::Impl {
    ClassifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
    std::vector<std::string> class_labels_;
};

Classifier::Classifier(const ClassifierConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Audio classifier engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);

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

std::vector<AudioClassificationResult> Classifier::predict(const std::vector<float>& waveform, int top_k) {
    if (!pimpl_) throw std::runtime_error("Classifier is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});
    const core::Tensor& logits_tensor = output_tensors[0];

    std::vector<float> logits(logits_tensor.num_elements());
    logits_tensor.copy_to_host(logits.data());

    std::vector<int> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::partial_sort(indices.begin(), indices.begin() + top_k, indices.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });

    float max_logit = logits[indices[0]];
    float sum_exp = 0.0f;
    std::vector<float> top_k_probs;
    for (int i = 0; i < top_k; ++i) {
        float prob = std::exp(logits[indices[i]] - max_logit);
        top_k_probs.push_back(prob);
        sum_exp += prob;
    }

    std::vector<AudioClassificationResult> results;
    for (int i = 0; i < top_k; ++i) {
        int class_id = indices[i];
        float confidence = top_k_probs[i] / sum_exp;
        std::string label = pimpl_->class_labels_.empty() || class_id >= pimpl_->class_labels_.size() ?
                            "Class " + std::to_string(class_id) :
                            pimpl_->class_labels_[class_id];
        results.push_back({class_id, confidence, label});
    }

    return results;
}

} // namespace xinfer::zoo::audio```