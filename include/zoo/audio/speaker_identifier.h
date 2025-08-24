#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <xinfer/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    using SpeakerEmbedding = std::vector<float>;

    struct SpeakerIdentificationResult {
        std::string speaker_label;
        float similarity_score;
    };

    struct SpeakerIdentifierConfig {
        std::string engine_path;
        preproc::AudioProcessorConfig audio_config;
    };

    class SpeakerIdentifier {
    public:
        explicit SpeakerIdentifier(const SpeakerIdentifierConfig& config);
        ~SpeakerIdentifier();

        SpeakerIdentifier(const SpeakerIdentifier&) = delete;
        SpeakerIdentifier& operator=(const SpeakerIdentifier&) = delete;
        SpeakerIdentifier(SpeakerIdentifier&&) noexcept;
        SpeakerIdentifier& operator=(SpeakerIdentifier&&) noexcept;

        void register_speaker(const std::string& label, const std::vector<float>& voice_sample);

        SpeakerIdentificationResult identify(const std::vector<float>& unknown_voice_sample);

        float compare(const SpeakerEmbedding& emb1, const SpeakerEmbedding& emb2);

    private:
        SpeakerEmbedding get_embedding(const std::vector<float>& waveform);

        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

