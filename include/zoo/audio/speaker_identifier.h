#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::audio {

    struct SpeakerResult {
        bool is_known;
        std::string speaker_id; // e.g., "user_1234"
        float confidence; // Cosine Similarity to best match
    };

    struct SpeakerIdConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ecapa_tdnn.onnx)
        // Expected Input: Mel Spectrogram
        // Expected Output: Speaker Embedding (e.g. 1x192)
        std::string model_path;

        // --- Audio Preprocessing ---
        int sample_rate = 16000;

        // Spectrogram settings
        int n_fft = 512;
        int hop_length = 160;
        int n_mels = 80;

        // --- Logic ---
        // Threshold to consider a voice a "match"
        float match_threshold = 0.75f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SpeakerIdentifier {
    public:
        explicit SpeakerIdentifier(const SpeakerIdConfig& config);
        ~SpeakerIdentifier();

        // Move semantics
        SpeakerIdentifier(SpeakerIdentifier&&) noexcept;
        SpeakerIdentifier& operator=(SpeakerIdentifier&&) noexcept;
        SpeakerIdentifier(const SpeakerIdentifier&) = delete;
        SpeakerIdentifier& operator=(const SpeakerIdentifier&) = delete;

        /**
         * @brief Create a "voiceprint" for a user and enroll them.
         *
         * @param speaker_id Unique ID for this person.
         * @param enrollment_audio A clean audio clip of the person speaking.
         */
        void enroll_speaker(const std::string& speaker_id, const std::vector<float>& enrollment_audio);

        /**
         * @brief Identify who is speaking in an audio clip.
         *
         * @param pcm_data Raw audio clip.
         * @return The ID of the best matching speaker.
         */
        SpeakerResult identify(const std::vector<float>& pcm_data);

        /**
         * @brief Clear the enrollment database.
         */
        void clear_database();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::audio