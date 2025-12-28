#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    /**
     * @brief A single separated audio stem.
     */
    struct AudioStem {
        std::string name; // "vocals", "drums", "bass", "other"
        std::vector<float> audio; // Raw PCM waveform
    };

    struct SeparatorConfig {
        // Hardware Target (GPU is strongly recommended for this task)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., spleeter_4stems.engine)
        // Expected Input: Spectrogram [1, 2, F, T] (Stereo)
        // Expected Output: Multiple Spectrogram Masks
        std::string model_path;

        // --- Audio Processing Specs ---
        int sample_rate = 44100; // High-quality audio

        // FFT parameters (Spleeter defaults)
        int n_fft = 4096;
        int hop_length = 1024;

        // Names of the output stems, in the order the model produces them
        std::vector<std::string> stem_names = {"vocals", "drums", "bass", "other"};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class MusicSourceSeparator {
    public:
        explicit MusicSourceSeparator(const SeparatorConfig& config);
        ~MusicSourceSeparator();

        // Move semantics
        MusicSourceSeparator(MusicSourceSeparator&&) noexcept;
        MusicSource-Separator& operator=(MusicSourceSeparator&&) noexcept;
        MusicSourceSeparator(const MusicSourceSeparator&) = delete;
        MusicSourceSeparator& operator=(const MusicSourceSeparator&) = delete;

        /**
         * @brief Separate a mixed audio track into its sources.
         *
         * Pipeline:
         * 1. Preprocess: STFT to create Magnitude & Phase Spectrograms.
         * 2. Inference: UNet predicts a mask for each stem.
         * 3. Postprocess: Apply masks to original magnitude and inverse STFT.
         *
         * @param stereo_pcm_data Interleaved stereo audio [L, R, L, R, ...].
         * @return A map of stem names to their waveforms.
         */
        std::map<std::string, std::vector<float>> separate(const std::vector<float>& stereo_pcm_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio