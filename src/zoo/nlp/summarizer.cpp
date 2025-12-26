#include <xinfer/zoo/nlp/summarizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/llm_sampler_interface.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <cstring>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Summarizer::Impl {
    SummarizerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> encoder_engine_;
    std::unique_ptr<backends::IBackend> decoder_engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::ILlmSampler> sampler_;

    // Data Containers
    // Encoder IO
    core::Tensor enc_input_ids;
    core::Tensor enc_attention_mask;
    core::Tensor enc_hidden_states; // Output of encoder, Input to decoder

    // Decoder IO
    core::Tensor dec_input_ids;     // [1, CurrentSeqLen]
    core::Tensor dec_logits;        // [1, VocabSize] (for last token)

    // State
    std::vector<int> history_ids_;
    int pad_token_id_ = 0;
    int eos_token_id_ = 1;
    int decoder_start_token_id_ = 0;

    Impl(const SummarizerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Engines
        encoder_engine_ = backends::BackendFactory::create(config_.target);
        decoder_engine_ = backends::BackendFactory::create(config_.target);

        // Load Encoder
        xinfer::Config enc_cfg; enc_cfg.model_path = config_.encoder_model_path;
        if (!encoder_engine_->load_model(enc_cfg.model_path)) {
            throw std::runtime_error("Summarizer: Failed to load encoder " + config_.encoder_model_path);
        }

        // Load Decoder
        xinfer::Config dec_cfg; dec_cfg.model_path = config_.decoder_model_path;
        if (!decoder_engine_->load_model(dec_cfg.model_path)) {
            throw std::runtime_error("Summarizer: Failed to load decoder " + config_.decoder_model_path);
        }

        // 2. Setup Tokenizer
        // T5 uses SentencePiece (BPE-like), BART uses GPT2-BPE
        auto type = config_.is_t5 ? preproc::text::TokenizerType::SENTENCEPIECE
                                  : preproc::text::TokenizerType::GPT_BPE;

        tokenizer_ = preproc::create_text_preprocessor(type, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.tokenizer_path;
        tok_cfg.max_length = config_.max_source_length;
        tokenizer_->init(tok_cfg);

        // Configure Special Tokens (T5 defaults)
        if (config_.is_t5) {
            pad_token_id_ = 0;
            eos_token_id_ = 1;
            decoder_start_token_id_ = 0; // T5 starts decoder with Pad
        } else {
            // BART defaults
            pad_token_id_ = 1;
            eos_token_id_ = 2;
            decoder_start_token_id_ = 2; // BART starts with EOS/BOS
        }

        // 3. Setup Sampler
        sampler_ = postproc::create_llm_sampler(config_.target);
        postproc::LlmSampleConfig samp_cfg;
        samp_cfg.temperature = config_.temperature;
        samp_cfg.top_k = 1; // Greedy by default for summarization accuracy
        samp_cfg.vocab_size = 32128; // T5 vocab size
        sampler_->init(samp_cfg);
    }

    std::string generate_summary(const std::string& text) {
        // --- Step 1: Encode Source ---
        // Tokenize
        tokenizer_->process(text, enc_input_ids, enc_attention_mask);

        // Run Encoder
        // Input: [Batch, SeqLen] -> Output: [Batch, SeqLen, Hidden]
        // This output is static for the entire generation loop.
        encoder_engine_->predict({enc_input_ids, enc_attention_mask}, {enc_hidden_states});

        // --- Step 2: Decode Loop ---
        history_ids_.clear();
        history_ids_.push_back(decoder_start_token_id_);

        for (int i = 0; i < config_.max_target_length; ++i) {
            // Update Decoder Input Tensor
            // Note: For optimal perf, use KV-Caching backend support.
            // Here we show the generic "Feed All History" approach.
            size_t curr_len = history_ids_.size();
            dec_input_ids.resize({1, (int64_t)curr_len}, core::DataType::kINT32);
            std::memcpy(dec_input_ids.data(), history_ids_.data(), curr_len * sizeof(int));

            // Run Decoder
            // Inputs: [DecoderIds, EncoderHiddenStates]
            // Note: Attention mask for encoder is often implicit or passed if backend requires it.
            decoder_engine_->predict({dec_input_ids, enc_hidden_states}, {dec_logits});

            // Sample Next Token
            // Sampler takes [1, VocabSize] logits for the last step
            std::vector<int> next_token_vec = sampler_->sample(dec_logits, dec_input_ids);
            int next_id = next_token_vec[0];

            // Stop Condition
            if (next_id == eos_token_id_) {
                break;
            }

            history_ids_.push_back(next_id);
        }

        // --- Step 3: Detokenize ---
        // Remove start token
        if (!history_ids_.empty()) history_ids_.erase(history_ids_.begin());

        core::Tensor output_ids({1, (int64_t)history_ids_.size()}, core::DataType::kINT32);
        std::memcpy(output_ids.data(), history_ids_.data(), history_ids_.size() * sizeof(int));

        return tokenizer_->decode(output_ids);
    }
};

// =================================================================================
// Public API
// =================================================================================

Summarizer::Summarizer(const SummarizerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Summarizer::~Summarizer() = default;
Summarizer::Summarizer(Summarizer&&) noexcept = default;
Summarizer& Summarizer::operator=(Summarizer&&) noexcept = default;

std::string Summarizer::summarize(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("Summarizer is null.");

    // Optional: Add prefix for T5 ("summarize: " + text) if model expects it
    std::string input_text = text;
    if (pimpl_->config_.is_t5) {
        input_text = "summarize: " + text;
    }

    return pimpl_->generate_summary(input_text);
}

} // namespace xinfer::zoo::nlp