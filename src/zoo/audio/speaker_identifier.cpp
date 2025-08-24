#include <include/zoo/audio/speaker_identifier.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

#include <include/core/engine.h>

namespace xinfer::zoo::audio {

struct SpeakerIdentifier::Impl {
    SpeakerIdentifierConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::AudioProcessor> preprocessor_;
    std::map<std::string, SpeakerEmbedding> known_speakers_;
};

SpeakerIdentifier::SpeakerIdentifier(const SpeakerIdentifierConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Speaker identifier engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    pimpl_->preprocessor_ = std::make_unique<preproc::AudioProcessor>(pimpl_->config_.audio_config);
}

SpeakerIdentifier::~SpeakerIdentifier() = default;
SpeakerIdentifier::SpeakerIdentifier(SpeakerIdentifier&&) noexcept = default;
SpeakerIdentifier& SpeakerIdentifier::operator=(SpeakerIdentifier&&) noexcept = default;

SpeakerEmbedding SpeakerIdentifier::get_embedding(const std::vector<float>& waveform) {
    if (!pimpl_) throw std::runtime_error("SpeakerIdentifier is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor spectrogram_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(waveform, spectrogram_tensor);

    auto output_tensors = pimpl_->engine_->infer({spectrogram_tensor});
    const core::Tensor& embedding_tensor = output_tensors[0];

    SpeakerEmbedding embedding(embedding_tensor.num_elements());
    embedding_tensor.copy_to_host(embedding.data());

    float norm = 0.0f;
    for (float val : embedding) {
        norm += val * val;
    }
    norm = std::sqrt(norm);

    if (norm > 1e-6) {
        for (float& val : embedding) {
            val /= norm;
        }
    }

    return embedding;
}

void SpeakerIdentifier::register_speaker(const std::string& label, const std::vector<float>& voice_sample) {
    pimpl_->known_speakers_[label] = get_embedding(voice_sample);
}

SpeakerIdentificationResult SpeakerIdentifier::identify(const std::vector<float>& unknown_voice_sample) {
    if (pimpl_->known_speakers_.empty()) {
        return {"Unknown", 0.0f};
    }

    SpeakerEmbedding unknown_embedding = get_embedding(unknown_voice_sample);

    float best_score = -1.0f;
    std::string best_label = "Unknown";

    for (const auto& pair : pimpl_->known_speakers_) {
        float score = compare(unknown_embedding, pair.second);
        if (score > best_score) {
            best_score = score;
            best_label = pair.first;
        }
    }

    return {best_label, best_score};
}

float SpeakerIdentifier::compare(const SpeakerEmbedding& emb1, const SpeakerEmbedding& emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) {
        return 0.0f;
    }
    float dot_product = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    return std::max(0.0f, std::min(1.0f, dot_product));
}

} // namespace xinfer::zoo::audio