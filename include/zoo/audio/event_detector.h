#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/preproc/audio_processor.h>

namespace xinfer::zoo::audio {

    struct AudioEvent {
        int class_id;
        float confidence;
        std::string label;
        float start_time_seconds;
        float end_time_seconds;
    };

    struct EventDetectorConfig {
        std::string engine_path;
        std::string labels_path = "";
        float event_threshold = 0.5f;
        preproc::AudioProcessorConfig audio_config;
    };

    class EventDetector {
    public:
        explicit EventDetector(const EventDetectorConfig& config);
        ~EventDetector();

        EventDetector(const EventDetector&) = delete;
        EventDetector& operator=(const EventDetector&) = delete;
        EventDetector(EventDetector&&) noexcept;
        EventDetector& operator=(EventDetector&&) noexcept;

        std::vector<AudioEvent> predict(const std::vector<float>& waveform);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio

