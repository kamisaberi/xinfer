#include <xinfer/zoo/accessibility/speech_augmenter.h>
#include <xinfer/core/logging.h>

// --- We reuse the Voice Converter module as the underlying engine ---
#include <xinfer/zoo/generative/voice_converter.h>

#include <iostream>
#include <vector>

namespace xinfer::zoo::accessibility {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SpeechAugmenter::Impl {
    AugmenterConfig config_;

    // The Voice Converter module handles the multi-stage pipeline
    std::unique_ptr<generative::VoiceConverter> vc_engine_;

    Impl(const AugmenterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the underlying Voice Converter with our specific models
        generative::VoiceConverterConfig vc_cfg;
        vc_cfg.target = config_.target;

        // Map Augmenter models to Voice Converter components
        vc_cfg.hubert_model_path = config_.encoder_model_path; // Content Encoder
        vc_cfg.generator_model_path = config_.vocoder_model_path; // Synthesizer
        vc_cfg.speaker_embedding_path = config_.target_voice_path; // The "clear" voice

        vc_cfg.sample_rate = config_.sample_rate;
        vc_cfg.vendor_params = config_.vendor_params;

        vc_engine_ = std::make_unique<generative::VoiceConverter>(vc_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

SpeechAugmenter::SpeechAugmenter(const AugmenterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SpeechAugmenter::~SpeechAugmenter() = default;
SpeechAugmenter::SpeechAugmenter(SpeechAugmenter&&) noexcept = default;
SpeechAugmenter& SpeechAugmenter::operator=(SpeechAugmenter&&) noexcept = default;

AugmenterResult SpeechAugmenter::process_chunk(const std::vector<float>& pcm_chunk) {
    if (!pimpl_ || !pimpl_->vc_engine_) {
        throw std::runtime_error("SpeechAugmenter is not initialized.");
    }

    // 1. Pad/trim chunk to the fixed size expected by the model
    size_t target_samples = (size_t)(pimpl_->config_.chunk_duration_sec * pimpl_->config_.sample_rate);
    std::vector<float> fixed_chunk(target_samples, 0.0f);

    size_t copy_len = std::min(pcm_chunk.size(), target_samples);
    std::copy(pcm_chunk.begin(), pcm_chunk.begin() + copy_len, fixed_chunk.begin());

    // 2. Delegate to the Voice Converter
    auto vc_res = pimpl_->vc_engine_->convert(fixed_chunk);

    // 3. Map to Augmenter result format
    AugmenterResult result;
    result.audio = vc_res.audio;
    result.sample_rate = vc_res.sample_rate;

    return result;
}

} // namespace xinfer::zoo::accessibility