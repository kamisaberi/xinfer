#include <xinfer/zoo/nlp/ner.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory unused; custom IOB decoding logic required.

#include <fstream>
#include <iostream>
#include <algorithm>
#include <map>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct NamedEntityRecognizer::Impl {
    NerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_logits; // [1, SeqLen, NumClasses]

    // Mapping: Index -> Label String
    std::vector<std::string> id2label_;

    Impl(const NerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("NER: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = config_.do_lower_case;
        tokenizer_->init(tok_cfg);

        // 3. Load Labels
        load_labels(config_.labels_path);
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            XINFER_LOG_ERROR("Could not open labels file: " + path);
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            id2label_.push_back(line);
        }
    }

    // --- IOB Decoding Logic ---
    std::vector<Entity> decode_logits() {
        std::vector<Entity> entities;

        auto shape = output_logits.shape();
        int seq_len = (int)shape[1];
        int num_classes = (int)shape[2];
        const float* logits = static_cast<const float*>(output_logits.data());

        // We need the input IDs to reconstruct the text
        const int32_t* ids_ptr = static_cast<const int32_t*>(input_ids.data());

        // State variables for aggregation
        Entity current_entity;
        current_entity.label = "";
        float current_score_sum = 0.0f;
        int current_token_count = 0;

        for (int i = 0; i < seq_len; ++i) {
            // 1. ArgMax
            const float* probs = logits + (i * num_classes);
            int max_id = 0;
            float max_val = probs[0];
            for (int c = 1; c < num_classes; ++c) {
                if (probs[c] > max_val) {
                    max_val = probs[c];
                    max_id = c;
                }
            }

            // 2. Decode Token String
            // We use a small temporary tensor to decode single ID
            core::Tensor single_id({1}, core::DataType::kINT32);
            ((int*)single_id.data())[0] = ids_ptr[i];
            std::string token_str = tokenizer_->decode(single_id);

            // Skip Special Tokens
            if (token_str == "[CLS]" || token_str == "[SEP]" || token_str == "[PAD]") {
                continue;
            }

            // 3. Get Tag (e.g. "B-PER", "I-PER", "O")
            std::string tag = (max_id < id2label_.size()) ? id2label_[max_id] : "O";

            // 4. State Machine (IOB)
            char prefix = (tag.size() > 0) ? tag[0] : 'O';
            std::string type = (tag.size() > 2) ? tag.substr(2) : "";

            // Check if we are continuing a subword (WordPiece "##")
            bool is_subword = (token_str.rfind("##", 0) == 0);
            if (is_subword) token_str = token_str.substr(2);

            bool start_new = false;
            bool append_current = false;

            if (prefix == 'B') {
                start_new = true;
            } else if (prefix == 'I') {
                if (current_entity.label == type) {
                    append_current = true;
                } else {
                    // "I-" without matching "B-" implies start of new entity in some schemes
                    start_new = true;
                }
            } else {
                // "O" -> End entity
                if (!current_entity.label.empty()) {
                    // Flush
                    current_entity.confidence = current_score_sum / current_token_count;
                    if (current_entity.confidence >= config_.min_confidence) {
                        entities.push_back(current_entity);
                    }
                    current_entity.label = ""; // Reset
                }
            }

            // Handle Transitions
            if (start_new) {
                // Flush old if exists
                if (!current_entity.label.empty()) {
                    current_entity.confidence = current_score_sum / current_token_count;
                    if (current_entity.confidence >= config_.min_confidence) {
                        entities.push_back(current_entity);
                    }
                }
                // Init new
                current_entity.text = token_str;
                current_entity.label = type;
                current_entity.start_char = 0; // TODO: Needs alignment mapping from tokenizer
                current_score_sum = max_val; // Note: Should probably softmax score first
                current_token_count = 1;
            } else if (append_current) {
                // Append
                if (!is_subword) current_entity.text += " ";
                current_entity.text += token_str;
                current_score_sum += max_val;
                current_token_count++;
            }
        }

        // Flush last
        if (!current_entity.label.empty()) {
            current_entity.confidence = current_score_sum / current_token_count;
            if (current_entity.confidence >= config_.min_confidence) {
                entities.push_back(current_entity);
            }
        }

        return entities;
    }
};

// =================================================================================
// Public API
// =================================================================================

NamedEntityRecognizer::NamedEntityRecognizer(const NerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

NamedEntityRecognizer::~NamedEntityRecognizer() = default;
NamedEntityRecognizer::NamedEntityRecognizer(NamedEntityRecognizer&&) noexcept = default;
NamedEntityRecognizer& NamedEntityRecognizer::operator=(NamedEntityRecognizer&&) noexcept = default;

std::vector<Entity> NamedEntityRecognizer::extract(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("NER is null.");

    // 1. Tokenize
    pimpl_->tokenizer_->process(text, pimpl_->input_ids, pimpl_->attention_mask);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_ids, pimpl_->attention_mask}, {pimpl_->output_logits});

    // 3. Decode
    return pimpl_->decode_logits();
}

} // namespace xinfer::zoo::nlp