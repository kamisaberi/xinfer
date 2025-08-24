#include <include/zoo/audio/language_identifier.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <cmath>

#include <include/core/engine.h>

namespace xinfer::zoo::audio {

struct LanguageIdentifier::Impl {
    LanguageIdentifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
    std::vector<std::string> language_labels_;
};

LanguageIdentifier::LanguageIdentifier(const LanguageIdentifierConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Language identifier engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->language_labels_.push_back(line);
        }
    }
}

LanguageIdentifier::~LanguageIdentifier() = default;
LanguageIdentifier::LanguageIdentifier(LanguageIdentifier&&) noexcept = default;
LanguageIdentifier& LanguageIdentifier::operator=(LanguageIdentifier&&) noexcept = default;

LanguageIdentificationResult LanguageIdentifier::predict(const std::vector<float>& waveform) {
    if (!pimpl_) throw std::runtime_error("LanguageIdentifier is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});
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

    LanguageIdentificationResult result;
    result.lang_id = max_idx;
    result.confidence = confidence;
    if (max_idx < pimpl_->language_labels_.size()) {
        result.language_code = pimpl_->language_labels_[max_idx];
    } else {
        result.language_code = "Unknown";
    }

    return result;
}

} // namespace xinfer::zoo::audio