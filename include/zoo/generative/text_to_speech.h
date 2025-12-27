#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct TtsResult {
        // Raw PCM audio samples (Float32, normalized [-1, 1])
        std::vector<float> audio;
        int sample_rate;
    };

    struct TtsConfig {
        // Hardware Target (GPU is strongly recommended for real-time)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // Acoustic Model (Text -> Mel Spectrogram)
        // e.g., tacotron2.engine
        std::string acoustic_model_path;

        // Vocoder Model (Mel -> Waveform)
        // e.g., hifigan.engine
        std::string vocoder_model_path;

        // --- Tokenizer/Frontend ---
        // For text-to-phoneme or text-to-ID conversion
        std::string vocab_path;

        // --- Audio Parameters ---
        int sample_rate = 22050; // Standard for many TTS models

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TextToSpeech {
    public:
        explicit TextToSpeech(const TtsConfig& config);
        ~TextToSpeech();

        // Move semantics
        TextToSpeech(TextToSpeech&&) noexcept;
        TextToSpeech& operator=(TextToSpeech&&) noexcept;
        TextToSpeech(const TextToSpeech&) = delete;
        TextToSpeech& operator=(const TextToSpeech&) = delete;

        /**
         * @brief Synthesize speech from text.
         *
         * @param text The input text.
         * @return TtsResult containing the audio waveform.
         */
        TtsResult synthesize(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative