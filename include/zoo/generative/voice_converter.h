#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::core { class Tensor; }

namespace xinfer::zoo::generative {

    using AudioWaveform = std::vector<float>;

    struct VoiceConverterConfig {
        std::string engine_path;
        // Add paths to speaker embedding files, vocabs, etc.
    };

    class VoiceConverter {
    public:
        explicit VoiceConverter(const VoiceConverterConfig& config);
        ~VoiceConverter();

        VoiceConverter(const VoiceConverter&) = delete;
        VoiceConverter& operator=(const VoiceConverter&) = delete;
        VoiceConverter(VoiceConverter&&) noexcept;
        VoiceConverter& operator=(VoiceConverter&&) noexcept;

        AudioWaveform predict(const AudioWaveform& source_audio, const AudioWaveform& target_voice_sample);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

