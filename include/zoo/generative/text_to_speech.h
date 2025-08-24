#pragma once


#include <string>
#include <vector>
#include <memory>

namespace xinfer::core { class Tensor; }

namespace xinfer::zoo::generative {

    using AudioWaveform = std::vector<float>;

    struct TextToSpeechConfig {
        std::string spectrogram_engine_path;
        std::string vocoder_engine_path;
        // Add paths to tokenizer vocabs, etc.
    };

    class TextToSpeech {
    public:
        explicit TextToSpeech(const TextToSpeechConfig& config);
        ~TextToSpeech();

        TextToSpeech(const TextToSpeech&) = delete;
        TextToSpeech& operator=(const TextToSpeech&) = delete;
        TextToSpeech(TextToSpeech&&) noexcept;
        TextToSpeech& operator=(TextToSpeech&&) noexcept;

        AudioWaveform predict(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

