#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    struct LanguageResult {
        std::string language_code; // "en", "es", "zh"
        std::string language_name; // "English", "Spanish", "Chinese"
        float confidence;
    };

    struct LanguageIdConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ecapa_tdnn_lid.onnx)
        std::string model_path;

        // Label Map (Class ID -> Language Name/Code)
        std::string labels_path;

        // --- Audio Preprocessing ---
        int sample_rate = 16000;
        float duration_sec = 3.0f; // Model expects fixed-length input

        // Spectrogram settings (must match training)
        int n_fft = 512;
        int hop_length = 160;
        int n_mels = 80;

        // --- Post-processing ---
        int top_k = 1;
        float confidence_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class LanguageIdentifier {
    public:
        explicit LanguageIdentifier(const LanguageIdConfig& config);
        ~LanguageIdentifier();

        // Move semantics
        LanguageIdentifier(LanguageIdentifier&&) noexcept;
        LanguageIdentifier& operator=(LanguageIdentifier&&) noexcept;
        LanguageIdentifier(const LanguageIdentifier&) = delete;
        LanguageIdentifier& operator=(const LanguageIdentifier&) = delete;

        /**
         * @brief Identify the language spoken in an audio clip.
         *
         * @param pcm_data Raw float audio samples.
         * @return List of top_k predicted languages.
         */
        std::vector<LanguageResult> identify(const std::vector<float>& pcm_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio