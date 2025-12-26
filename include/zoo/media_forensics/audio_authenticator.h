#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::media_forensics {

    /**
     * @brief Result of Audio Forensics analysis.
     */
    struct AuthResult {
        bool is_deepfake;       // True if fake_score > threshold
        float fake_score;       // Probability [0.0 - 1.0]
        float real_score;
        std::string label;      // "Real", "Fake", "Unknown"
        float confidence;
    };

    struct AudioAuthConfig {
        // Hardware Target (CPU is usually sufficient for audio, NPU for batching)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., rawnet3_anti_spoof.onnx)
        std::string model_path;

        // Input Audio Specs
        int sample_rate = 16000; // Standard for telephony/forensics
        int duration_sec = 3;    // Crop/Pad input to this length

        // Feature Extractor (Mel-Spectrogram vs Raw Waveform)
        // If false, passes raw PCM to model (e.g. RawNet).
        // If true, converts to Spectrogram first (e.g. ResNet).
        bool use_spectrogram = true;
        int n_fft = 1024;
        int n_mels = 80;

        // Sensitivity
        float threshold = 0.5f;  // Score > 0.5 implies Fake

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class AudioAuthenticator {
    public:
        explicit AudioAuthenticator(const AudioAuthConfig& config);
        ~AudioAuthenticator();

        // Move semantics
        AudioAuthenticator(AudioAuthenticator&&) noexcept;
        AudioAuthenticator& operator=(AudioAuthenticator&&) noexcept;
        AudioAuthenticator(const AudioAuthenticator&) = delete;
        AudioAuthenticator& operator=(const AudioAuthenticator&) = delete;

        /**
         * @brief Authenticate an audio clip.
         *
         * Pipeline:
         * 1. Resample/Crop audio to fixed length.
         * 2. (Optional) Convert to Mel-Spectrogram.
         * 3. Inference (Binary Classification).
         *
         * @param pcm_data Raw float audio samples (normalized -1.0 to 1.0).
         * @return Authenticity result.
         */
        AuthResult authenticate(const std::vector<float>& pcm_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::media_forensics