#include <xinfer/zoo/generative/text_to_speech.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h> // For TextTokenizer
// Postproc is custom (Vocoder)

#include <iostream>
#include <vector>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TextToSpeech::Impl {
    TtsConfig config_;

    // --- Components ---
    std::unique_ptr<backends::IBackend> acoustic_engine_;
    std::unique_ptr<backends::IBackend> vocoder_engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // --- Tensors ---
    // Acoustic Model
    core::Tensor text_ids, mel_output;

    // Vocoder
    core::Tensor vocoder_output;

    Impl(const TtsConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Engines
        acoustic_engine_ = backends::BackendFactory::create(config_.target);
        vocoder_engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config ac_cfg; ac_cfg.model_path = config_.acoustic_model_path;
        if (!acoustic_engine_->load_model(ac_cfg.model_path))
            throw std::runtime_error("TTS: Failed to load acoustic model.");

        xinfer::Config vc_cfg; vc_cfg.model_path = config_.vocoder_model_path;
        if (!vocoder_engine_->load_model(vc_cfg.model_path))
            throw std::runtime_error("TTS: Failed to load vocoder model.");

        // 2. Setup Tokenizer
        // This is a simple text-to-ID tokenizer.
        // A full TTS frontend would also do Text Normalization and G2P.
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::WHITESPACE, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        // Max length depends on the acoustic model's capability
        tok_cfg.max_length = 256;
        tokenizer_->init(tok_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

TextToSpeech::TextToSpeech(const TtsConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TextToSpeech::~TextToSpeech() = default;
TextToSpeech::TextToSpeech(TextToSpeech&&) noexcept = default;
TextToSpeech& TextToSpeech::operator=(TextToSpeech&&) noexcept = default;

TtsResult TextToSpeech::synthesize(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("TextToSpeech is null.");

    TtsResult result;
    result.sample_rate = pimpl_->config_.sample_rate;

    // --- Step 1: Text Frontend (Tokenize) ---
    // In a real system, this would be: Text -> Numbers -> Phonemes
    // Here we just map characters to IDs via the tokenizer.

    core::Tensor text_mask; // Dummy
    pimpl_->tokenizer_->process(text, pimpl_->text_ids, text_mask);

    // --- Step 2: Acoustic Model (Text -> Mel Spectrogram) ---
    // Tacotron2/TransformerTTS are autoregressive.
    // We simulate a simplified feed-forward version here for clarity.
    // Input: [TextIDs] -> Output: [1, Mels, Time] Mel Spectrogram

    pimpl_->acoustic_engine_->predict({pimpl_->text_ids}, {pimpl_->mel_output});

    // --- Step 3: Vocoder (Mel -> Waveform) ---
    // HiFi-GAN/WaveGlow are fully-convolutional, very fast on GPU.
    // Input: [1, Mels, Time] -> Output: [1, 1, Samples]

    pimpl_->vocoder_engine_->predict({pimpl_->mel_output}, {pimpl_->vocoder_output});

    // --- Step 4: Format Output ---
    const float* ptr = static_cast<const float*>(pimpl_->vocoder_output.data());
    size_t count = pimpl_->vocoder_output.size();

    result.audio.assign(ptr, ptr + count);

    return result;
}

} // namespace xinfer::zoo::generative