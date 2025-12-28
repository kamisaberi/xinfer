#include <xinfer/zoo/audio/speaker_identifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SpeakerIdentifier::Impl {
    SpeakerIdConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IAudioPreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // --- Voiceprint Database ---
    // Maps Speaker ID -> Normalized Feature Vector
    std::map<std::string, std::vector<float>> database_;

    Impl(const SpeakerIdConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SpeakerIdentifier: Failed to load model.");
        }

        // 2. Setup Audio Preprocessor
        preproc_ = preproc::create_audio_preprocessor(config_.target);

        preproc::AudioPreprocConfig aud_cfg;
        aud_cfg.sample_rate = config_.sample_rate;
        aud_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
        aud_cfg.n_fft = config_.n_fft;
        aud_cfg.hop_length = config_.hop_length;
        aud_cfg.n_mels = config_.n_mels;
        aud_cfg.log_mel = true;

        preproc_->init(aud_cfg);
    }

    // --- Helper: Get Normalized Embedding ---
    std::vector<float> get_embedding(const std::vector<float>& pcm) {
        // 1. Preprocess
        preproc::AudioBuffer buf{pcm.data(), pcm.size(), config_.sample_rate};
        preproc_->process(buf, input_tensor);

        // 2. Inference
        engine_->predict({input_tensor}, {output_tensor});

        // 3. Normalize (L2)
        size_t count = output_tensor.size();
        const float* ptr = static_cast<const float*>(output_tensor.data());
        std::vector<float> vec(ptr, ptr + count);

        float sum_sq = 0.0f;
        for (float val : vec) sum_sq += val * val;
        float inv_norm = 1.0f / (std::sqrt(sum_sq) + 1e-9f);
        for (float& val : vec) val *= inv_norm;

        return vec;
    }

    // --- Helper: Cosine Similarity ---
    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || a.size() != b.size()) return 0.0f;
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
        return dot;
    }
};

// =================================================================================
// Public API
// =================================================================================

SpeakerIdentifier::SpeakerIdentifier(const SpeakerIdConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SpeakerIdentifier::~SpeakerIdentifier() = default;
SpeakerIdentifier::SpeakerIdentifier(SpeakerIdentifier&&) noexcept = default;
SpeakerIdentifier& SpeakerIdentifier::operator=(SpeakerIdentifier&&) noexcept = default;

void SpeakerIdentifier::clear_database() {
    if (pimpl_) pimpl_->database_.clear();
}

void SpeakerIdentifier::enroll_speaker(const std::string& speaker_id, const std::vector<float>& enrollment_audio) {
    if (!pimpl_) throw std::runtime_error("SpeakerIdentifier is null.");

    // In a real system, you might average embeddings from multiple clips.
    // Here, we take one.
    std::vector<float> embedding = pimpl_->get_embedding(enrollment_audio);
    pimpl_->database_[speaker_id] = embedding;

    XINFER_LOG_INFO("Enrolled speaker: " + speaker_id);
}

SpeakerResult SpeakerIdentifier::identify(const std::vector<float>& pcm_data) {
    if (!pimpl_) throw std::runtime_error("SpeakerIdentifier is null.");

    // 1. Get embedding for the query clip
    std::vector<float> query_vec = pimpl_->get_embedding(pcm_data);

    // 2. Search Database
    float best_score = -1.0f;
    std::string best_match_id = "Unknown";

    for (const auto& kv : pimpl_->database_) {
        float score = pimpl_->cosine_similarity(query_vec, kv.second);
        if (score > best_score) {
            best_score = score;
            best_match_id = kv.first;
        }
    }

    // 3. Format Result
    SpeakerResult result;
    result.confidence = best_score;

    if (best_score >= pimpl_->config_.match_threshold) {
        result.is_known = true;
        result.speaker_id = best_match_id;
    } else {
        result.is_known = false;
        result.speaker_id = "Unknown";
    }

    return result;
}

} // namespace xinfer::zoo::audio