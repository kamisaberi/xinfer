#include <xinfer/zoo/nlp/keyword_extractor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory unused; custom grouping logic is implemented here.

#include <iostream>
#include <algorithm>
#include <map>
#include <numeric>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct KeywordExtractor::Impl {
    KeywordConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_logits; // [1, SeqLen, 3]

    Impl(const KeywordConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("KeywordExtractor: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = config_.do_lower_case;
        tokenizer_->init(tok_cfg);
    }

    // --- Helper: Decode ID sequence to String ---
    std::string decode_ids(const std::vector<int>& ids) {
        if (ids.empty()) return "";

        // Wrap in Tensor for the tokenizer's decode method
        core::Tensor t_ids({1, (int64_t)ids.size()}, core::DataType::kINT32);
        std::memcpy(t_ids.data(), ids.data(), ids.size() * sizeof(int));

        return tokenizer_->decode(t_ids);
    }

    // --- Core Logic ---
    std::vector<Keyword> process_logits() {
        std::vector<Keyword> keywords;

        auto shape = output_logits.shape(); // [1, SeqLen, NumClasses]
        int seq_len = (int)shape[1];
        int num_classes = (int)shape[2];
        const float* logits = static_cast<const float*>(output_logits.data());

        // Access Input IDs to reconstruct text
        const int* ids_ptr = static_cast<const int*>(input_ids.data());

        std::vector<int> current_phrase_ids;
        float current_score_sum = 0.0f;
        int phrase_start_idx = -1;

        // Iterate tokens (skip [CLS], stop at [SEP] or len)
        for (int i = 0; i < seq_len; ++i) {
            int token_id = ids_ptr[i];

            // Check for padding/special tokens (0 is usually PAD in BERT)
            if (token_id == 0) break; // End of valid sequence
            if (token_id == 101 || token_id == 102) continue; // Skip CLS/SEP (standard IDs)

            // ArgMax
            const float* probs = logits + (i * num_classes);
            int max_id = 0;
            float max_val = probs[0];
            for (int c = 1; c < num_classes; ++c) {
                if (probs[c] > max_val) {
                    max_val = probs[c];
                    max_id = c;
                }
            }
            // Optional: Apply Softmax if model outputs raw logits to get true confidence
            // Here we just use the raw logit magnitude or simple check

            bool is_b = (max_id == config_.idx_b);
            bool is_i = (max_id == config_.idx_i);

            // Logic: Group B followed by Is
            if (is_b) {
                // If we were building a phrase, flush it
                if (!current_phrase_ids.empty()) {
                    keywords.push_back({decode_ids(current_phrase_ids), current_score_sum / current_phrase_ids.size(), phrase_start_idx, i});
                }
                // Start new
                current_phrase_ids = {token_id};
                current_score_sum = max_val;
                phrase_start_idx = i;
            }
            else if (is_i) {
                if (!current_phrase_ids.empty()) {
                    current_phrase_ids.push_back(token_id);
                    current_score_sum += max_val;
                } else {
                    // "I" without "B" usually implies continuation or weak start.
                    // Strict mode: Ignore. Loose mode: Treat as B.
                    // Treating as new phrase here:
                    current_phrase_ids = {token_id};
                    current_score_sum = max_val;
                    phrase_start_idx = i;
                }
            }
            else { // is_o
                // Flush current
                if (!current_phrase_ids.empty()) {
                    keywords.push_back({decode_ids(current_phrase_ids), current_score_sum / current_phrase_ids.size(), phrase_start_idx, i});
                    current_phrase_ids.clear();
                }
            }
        }

        // Flush final
        if (!current_phrase_ids.empty()) {
            keywords.push_back({decode_ids(current_phrase_ids), current_score_sum / current_phrase_ids.size(), phrase_start_idx, seq_len});
        }

        return keywords;
    }
};

// =================================================================================
// Public API
// =================================================================================

KeywordExtractor::KeywordExtractor(const KeywordConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

KeywordExtractor::~KeywordExtractor() = default;
KeywordExtractor::KeywordExtractor(KeywordExtractor&&) noexcept = default;
KeywordExtractor& KeywordExtractor::operator=(KeywordExtractor&&) noexcept = default;

std::vector<Keyword> KeywordExtractor::extract(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("KeywordExtractor is null.");

    // 1. Tokenize
    pimpl_->tokenizer_->process(text, pimpl_->input_ids, pimpl_->attention_mask);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_ids, pimpl_->attention_mask}, {pimpl_->output_logits});

    // 3. Post-process (Group Tags)
    auto raw_keywords = pimpl_->process_logits();

    // 4. Filtering & Deduplication
    std::vector<Keyword> filtered;
    std::vector<std::string> seen;

    for (auto& kw : raw_keywords) {
        if (kw.score < pimpl_->config_.min_score) continue;

        // Clean up string (remove surrounding spaces if tokenizer added them)
        if (!kw.phrase.empty() && kw.phrase[0] == ' ') kw.phrase = kw.phrase.substr(1);
        if (kw.phrase.length() < 2) continue; // Skip single chars

        if (pimpl_->config_.deduplicate) {
            std::string key = kw.phrase;
            // Simple lowercase check
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);

            bool exists = false;
            for(const auto& s : seen) if(s == key) exists = true;

            if (exists) continue;
            seen.push_back(key);
        }

        filtered.push_back(kw);
    }

    return filtered;
}

} // namespace xinfer::zoo::nlp