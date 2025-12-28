#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::accessibility {

    struct AugmenterResult {
        // Clear, synthesized speech
        std::vector<float> clear_audio;
        int sample_rate;
    };

    struct AugmenterConfig {
        // Hardware Target (Real-time requires GPU/NPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // 1. Content Encoder (e.g., modified HuBERT or Wav2Vec2)
        std::string encoder_model_path;

        // 2. Synthesizer/Vocoder (e.g., HiFi-GAN)
        std::string vocoder_model_path;

        // --- Audio Specs ---
        int sample_rate = 16000; // Models usually work at 16kHz or 22.05kHz
        float chunk_duration_sec = 1.0f; // Process audio in 1-second chunks

        // Optional: Target voice embedding if the model supports it
        std::string target_voice_path;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SpeechAugmenter {
    public:
        explicit SpeechAugmenter(const AugmenterConfig& config);
        ~SpeechAugmenter();

        // Move semantics
        SpeechAugmenter(SpeechAugmenter&&) noexcept;
        SpeechAugmenter& operator=(SpeechAugmenter&&) noexcept;
        SpeechAugmenter(const SpeechAugmenter&) = delete;
        SpeechAugmenter& operator=(const SpeechAugmenter&) = delete;

        /**
         * @brief Process a chunk of impaired speech.
         *
         * @param pcm_chunk Raw float audio from microphone.
         * @return Synthesized, clear audio.
         */
        AugmenterResult process_chunk(const std::vector<float>& pcm_chunk);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::accessibility