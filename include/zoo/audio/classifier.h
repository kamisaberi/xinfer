#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    struct AudioClassResult {
        int id;
        std::string label;
        float confidence;
    };

    struct AudioClassifierConfig {
        // Hardware Target (CPU/NPU usually sufficient for audio)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yamnet.onnx, panns_cnn.rknn)
        std::string model_path;

        // Label map (Class ID -> Event Name)
        std::string labels_path;

        // --- Audio Preprocessing ---
        int sample_rate = 16000;
        float duration_sec = 2.0f; // Model expects fixed-length input

        // Spectrogram settings (must match training)
        int n_fft = 1024;
        int hop_length = 320;
        int n_mels = 64;

        // --- Post-processing ---
        int top_k = 3;
        float confidence_threshold = 0.3f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Classifier {
    public:
        explicit Classifier(const AudioClassifierConfig& config);
        ~Classifier();

        // Move semantics
        Classifier(Classifier&&) noexcept;
        Classifier& operator=(Classifier&&) noexcept;
        Classifier(const Classifier&) = delete;
        Classifier& operator=(const Classifier&) = delete;

        /**
         * @brief Classify an audio clip.
         *
         * @param pcm_data Raw float audio samples (normalized [-1, 1]).
         * @return List of top_k predicted classes.
         */
        std::vector<AudioClassResult> classify(const std::vector<float>& pcm_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio