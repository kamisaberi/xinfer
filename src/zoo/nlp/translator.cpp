#include <xinfer/zoo/nlp/translator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/text/llm_sampler_interface.h>

#include <iostream>
#include <vector>
#include <cstring>
#include <algorithm>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Translator::Impl {
    TranslatorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> encoder_engine_;
    std::unique_ptr<backends::IBackend> decoder_engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::ILlmSampler> sampler_;

    // Data Containers
    core::Tensor enc_input_ids, enc_attention_mask, enc_output;
    core::Tensor dec_input_ids, dec_logits;

    // Language IDs
    int src_lang_id_ = -1;
    int tgt_lang_id_ = -1;
    int eos_token_id_ = 2; // Default (often 1 or 2)

    // Helper buffer
    std::vector<int> history_ids_;

    Impl(const TranslatorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backends
        encoder_engine_ = backends::BackendFactory::create(config_.target);
        decoder_engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config enc_cfg, dec_cfg;
        enc_cfg.model_path = config_.encoder_path;
        dec_cfg.model_path = config_.decoder_path;

        if (!encoder_engine_->load_model(enc_cfg.model_path))
            throw std::runtime_error("Translator: Failed to load encoder.");
        if (!decoder_engine_->load_model(dec_cfg.model_path))
            throw std::runtime_error("Translator: Failed to load decoder.");

        // 2. Setup Tokenizer
        // NLLB/Marian use SentencePiece
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::SENTENCEPIECE, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.tokenizer_path;
        tok_cfg.max_length = config_.max_source_length;
        tokenizer_->init(tok_cfg);

        // 3. Setup Sampler
        sampler_ = postproc::create_llm_sampler(config_.target);
        postproc::LlmSampleConfig samp_cfg;
        samp_cfg.temperature = 1.0f; // Greedy is usually best for translation
        samp_cfg.top_k = 1;
        samp_cfg.vocab_size = 128112; // NLLB-200 vocab size, adjust for model
        sampler_->init(samp_cfg);

        // 4. Resolve Initial Language IDs
        resolve_language_ids(config_.src_lang, config_.tgt_lang);
    }

    bool resolve_language_ids(const std::string& src, const std::string& tgt) {
        // We use the tokenizer to find the ID of the special language tokens
        // e.g. "eng_Latn" -> 256047

        auto lookup_id = [&](const std::string& token) -> int {
            // Hack: Encode the special token and see what ID it gets
            // Ideally, ITextPreprocessor should expose a `get_id(string)` method.
            // Using `process` for now.
            core::Tensor tmp_ids, tmp_mask;
            tokenizer_->process(token, tmp_ids, tmp_mask);
            int* ptr = (int*)tmp_ids.data();
            // Usually returns [BOS, LANG_ID, EOS] or just [LANG_ID] depending on tokenizer config
            // Simple heuristic: pick the first ID that isn't 0, 1, or 2 if possible
            for(int i=0; i<tmp_ids.shape()[1]; ++i) {
                if (ptr[i] > 3) return ptr[i];
            }
            return -1;
        };

        int s_id = lookup_id(src);
        int t_id = lookup_id(tgt);

        if (s_id == -1 || t_id == -1) {
            XINFER_LOG_ERROR("Could not resolve language tokens for " + src + " -> " + tgt);
            return false;
        }

        src_lang_id_ = s_id;
        tgt_lang_id_ = t_id;
        XINFER_LOG_INFO("Languages set: " + src + "(" + std::to_string(s_id) + ") -> " + tgt + "(" + std::to_string(t_id) + ")");
        return true;
    }

    std::string run_translation(const std::string& text) {
        if (src_lang_id_ == -1 || tgt_lang_id_ == -1) return "[Error: Lang ID unset]";

        // --- Step 1: Encoder ---
        // Tokenize
        tokenizer_->process(text, enc_input_ids, enc_attention_mask);

        // Prepend Source Lang ID if model requires it (NLLB does not, it uses it for decoder start. M2M might).
        // Standard NLLB: Encoder input is just text. Decoder input starts with Target Lang.

        // Execute Encoder
        // Input: [Batch, Seq] -> Output: [Batch, Seq, Hidden]
        encoder_engine_->predict({enc_input_ids, enc_attention_mask}, {enc_output});

        // --- Step 2: Decoder ---
        // Initialize Decoder Input with Target Language ID (Forced BOS)
        history_ids_.clear();
        history_ids_.push_back(tgt_lang_id_); // <--- Critical for NMT

        for (int i = 0; i < config_.max_target_length; ++i) {
            // Update Decoder Input
            size_t curr_len = history_ids_.size();
            dec_input_ids.resize({1, (int64_t)curr_len}, core::DataType::kINT32);
            std::memcpy(dec_input_ids.data(), history_ids_.data(), curr_len * sizeof(int));

            // Inference
            // Inputs: [DecoderIds, EncoderHiddenState, EncoderMask]
            decoder_engine_->predict({dec_input_ids, enc_output, enc_attention_mask}, {dec_logits});

            // Sample Next Token
            // Sampler handles extracting the last logit row
            std::vector<int> next_vec = sampler_->sample(dec_logits, dec_input_ids);
            int next_id = next_vec[0];

            if (next_id == eos_token_id_) break;

            history_ids_.push_back(next_id);
        }

        // --- Step 3: Detokenize ---
        // Remove the forced language token at the start
        if (!history_ids_.empty()) history_ids_.erase(history_ids_.begin());

        core::Tensor output_ids({1, (int64_t)history_ids_.size()}, core::DataType::kINT32);
        std::memcpy(output_ids.data(), history_ids_.data(), history_ids_.size() * sizeof(int));

        return tokenizer_->decode(output_ids);
    }
};

// =================================================================================
// Public API
// =================================================================================

Translator::Translator(const TranslatorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Translator::~Translator() = default;
Translator::Translator(Translator&&) noexcept = default;
Translator& Translator::operator=(Translator&&) noexcept = default;

bool Translator::set_languages(const std::string& src, const std::string& tgt) {
    if (!pimpl_) return false;
    return pimpl_->resolve_language_ids(src, tgt);
}

std::string Translator::translate(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("Translator is null.");
    return pimpl_->run_translation(text);
}

} // namespace xinfer::zoo::nlp