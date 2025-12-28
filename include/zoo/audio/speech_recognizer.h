#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    struct SpeechRecognizerConfig {
        // Hardware Target (CPU is fine for single stream, NPU/GPU for batch)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., quartznet.onnx)
        std::string model_path;

        // --- Audio Preprocessing ---
        int sample_rate = 16000;

        // Spectrogram settings
        int n_fft = 512;
        int hop_length = 160;
        int n_mels = 64;

        // --- CTC Decoder ---
        // Character map. e.g., "_' abcdefghijklmnopqrstuvwxyz"
        std::string vocabulary;
        int blank_index = 28; // Usually last char

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SpeechRecognizer {
    public:
        explicit SpeechRecognizer(const SpeechRecognizerConfig& config);
        ~SpeechRecognizer();

        // Move semantics
        SpeechRecognizer(SpeechRecognizer&&) noexcept;
        SpeechRecognizer& operator=(SpeechRecognizer&&) noexcept;
        SpeechRecognizer(const SpeechRecognizer&) = delete;
        SpeechRecognizer& operator=(const SpeechRecognizer&) = delete;

        /**
         * @brief Transcribe a full audio clip.
         *
         * @param pcm_data Raw float audio samples (normalized [-1, 1]).
         * @return The transcribed text for each batch item (usually one).
         */
        std::vector<std::string> recognize(const std::vector<float>& pcm_data);

        /**
         * @brief Transcribe a batch of audio clips.
         * @note Requires model to support batching. All clips should be padded
         *       to the same length.
         */
        std::vector<std::string> recognize_batch(const std::vector<std::vector<float>>& pcm_batch);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio