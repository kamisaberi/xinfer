#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    struct LanguageIdentificationResult {
        int lang_id;
        float confidence;
        std::string language_code; // e.g., "en", "es", "fr"
    };

    struct LanguageIdentifierConfig {
        std::string engine_path;
        std::string labels_path = "";
        preproc::AudioProcessorConfig audio_config;
    };

    class LanguageIdentifier {
    public:
        explicit LanguageIdentifier(const LanguageIdentifierConfig& config);
        ~LanguageIdentifier();

        LanguageIdentifier(const LanguageIdentifier&) = delete;
        LanguageIdentifier& operator=(const LanguageIdentifier&) = delete;
        LanguageIdentifier(LanguageIdentifier&&) noexcept;
        LanguageIdentifier& operator=(LanguageIdentifier&&) noexcept;

        LanguageIdentificationResult predict(const std::vector<float>& waveform);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

