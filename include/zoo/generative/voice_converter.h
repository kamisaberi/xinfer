#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct VoiceResult {
        // Raw PCM audio samples of the converted voice
        std::vector<float> audio;
        int sample_rate;
    };

    struct VoiceConverterConfig {
        // Hardware Target (GPU required for real-time performance)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // RVC is a multi-model system
        std::string hubert_model_path;      // Content Encoder
        std::string generator_model_path;   // Synthesizer
        std::string speaker_embedding_path; // .npy file for target voice

        // --- Audio Specs ---
        int sample_rate = 44100; // High-quality audio
        int chunk_size = 16384;  // Process audio in chunks

        // --- RVC Parameters ---
        int target_speaker_id = 0;
        float pitch_change = 0.0f; // Semitones to shift pitch

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class VoiceConverter {
    public:
        explicit VoiceConverter(const VoiceConverterConfig& config);
        ~VoiceConverter();

        // Move semantics
        VoiceConverter(VoiceConverter&&) noexcept;
        VoiceConverter& operator=(VoiceConverter&&) noexcept;
        VoiceConverter(const VoiceConverter&) = delete;
        VoiceConverter& operator=(const VoiceConverter&) = delete;

        /**
         * @brief Convert a chunk of audio in real-time.
         *
         * @param pcm_chunk A buffer of raw float audio.
         * @return The converted audio chunk.
         */
        VoiceResult convert(const std::vector<float>& pcm_chunk);

        /**
         * @brief Reset internal state (e.g. history buffers).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative