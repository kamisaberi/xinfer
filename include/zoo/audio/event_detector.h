#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    struct AudioEvent {
        std::string label;   // "Glass Break", "Dog Bark"
        float confidence;
        long long start_time_ms;
        long long end_time_ms;
    };

    struct EventDetectorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yamnet.onnx)
        std::string model_path;
        std::string labels_path;

        // --- Audio Specs ---
        int sample_rate = 16000;
        float window_sec = 0.96f;  // Model input duration (YAMNet is ~1s)
        float stride_sec = 0.48f;  // How often to run inference

        // --- Spectrogram settings ---
        int n_fft = 1024;
        int hop_length = 320;
        int n_mels = 64;

        // --- VAD (Voice Activity Detection) ---
        // Energy threshold to trigger analysis (prevents running on silence)
        float vad_energy_threshold = 0.005f;

        // --- Post-processing ---
        float confidence_threshold = 0.3f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class EventDetector {
    public:
        explicit EventDetector(const EventDetectorConfig& config);
        ~EventDetector();

        // Move semantics
        EventDetector(EventDetector&&) noexcept;
        EventDetector& operator=(EventDetector&&) noexcept;
        EventDetector(const EventDetector&) = delete;
        EventDetector& operator=(const EventDetector&) = delete;

        /**
         * @brief Process a continuous stream of audio data.
         *
         * @param pcm_chunk A small chunk of new audio from a microphone.
         * @param timestamp_ms The starting timestamp of this chunk.
         * @return A list of any new events detected within this chunk.
         */
        std::vector<AudioEvent> process_stream(const std::vector<float>& pcm_chunk, long long timestamp_ms);

        /**
         * @brief Reset all internal buffers.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio