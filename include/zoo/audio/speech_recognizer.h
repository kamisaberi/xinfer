#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    struct TranscriptionResult {
        std::string text;
        float confidence;
    };

    struct SpeechRecognizerConfig {
        std::string engine_path;
        std::string character_map_path;
        preproc::AudioProcessorConfig audio_config;
    };

    class SpeechRecognizer {
    public:
        explicit SpeechRecognizer(const SpeechRecognizerConfig& config);
        ~SpeechRecognizer();

        SpeechRecognizer(const SpeechRecognizer&) = delete;
        SpeechRecognizer& operator=(const SpeechRecognizer&) = delete;
        SpeechRecognizer(SpeechRecognizer&&) noexcept;
        SpeechRecognizer& operator=(SpeechRecognizer&&) noexcept;

        TranscriptionResult predict(const std::vector<float>& waveform);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

