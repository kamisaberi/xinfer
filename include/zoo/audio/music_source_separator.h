#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <include/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    using AudioWaveform = std::vector<float>;

    struct MusicSourceSeparatorConfig {
        std::string engine_path;
        preproc::AudioProcessorConfig audio_config;
        std::vector<std::string> source_names = {"drums", "bass", "other", "vocals"};
    };

    class MusicSourceSeparator {
    public:
        explicit MusicSourceSeparator(const MusicSourceSeparatorConfig& config);
        ~MusicSourceSeparator();

        MusicSourceSeparator(const MusicSourceSeparator&) = delete;
        MusicSourceSeparator& operator=(const MusicSourceSeparator&) = delete;
        MusicSourceSeparator(MusicSourceSeparator&&) noexcept;
        MusicSourceSeparator& operator=(MusicSourceSeparator&&) noexcept;

        std::map<std::string, AudioWaveform> predict(const std::vector<float>& mixed_waveform);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

