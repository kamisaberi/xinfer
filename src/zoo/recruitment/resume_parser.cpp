#include <xinfer/zoo/recruitment/resume_parser.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; NER decoding logic is specific text manipulation.

#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>

namespace xinfer::zoo::recruitment {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ResumeParser::Impl {
    ParserConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_logits; // [1, SeqLen, NumClasses]

    // Label Map (ID -> String)
    std::vector<std::string> id_to_label_;

    Impl(const ParserConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ResumeParser: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);
        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_seq_length;
        tok_cfg.do_lower_case = true; // Most NER models are uncased
        tokenizer_->init(tok_cfg);

        // 3. Load Labels
        load_labels(config_.labels_path);
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            id_to_label_.push_back(line);
        }
    }

    // --- Core Logic: Process one chunk of text ---
    std::vector<ResumeEntity> process_chunk(const std::string& text_chunk) {
        std::vector<ResumeEntity> entities;

        // 1. Tokenize
        tokenizer_->process(text_chunk, input_ids, attention_mask);

        // 2. Inference
        engine_->predict({input_ids, attention_mask}, {output_logits});

        // 3. Decode Logits -> Tags
        auto shape = output_logits.shape();
        int seq_len = (int)shape[1];
        int num_classes = (int)shape[2];
        const float* logits = static_cast<const float*>(output_logits.data());

        // Get Input IDs to reconstruct text
        const int* ids_ptr = static_cast<const int*>(input_ids.data());

        // Temporary buffer for current entity reconstruction
        std::string current_word;
        std::string current_tag = "O";
        float current_score = 0.0f;
        int tokens_in_entity = 0;

        for (int i = 0; i < seq_len; ++i) {
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

            // Decode Token String
            // We need a helper to get string from ID.
            // In a real app, tokenizer exposes `id_to_token`.
            // Here we use a simplified decode provided by ITextPreprocessor for the single ID.
            core::Tensor single_id({1}, core::DataType::kINT32);
            ((int*)single_id.data())[0] = ids_ptr[i];
            std::string token_str = tokenizer_->decode(single_id);

            // Skip Special Tokens
            if (token_str == "[CLS]" || token_str == "[SEP]" || token_str == "[PAD]") continue;

            std::string label = (max_id < id_to_label_.size()) ? id_to_label_[max_id] : "O";

            // IOB Logic (Inside-Outside-Beginning)
            // If tag starts with B-, it's a new entity.
            // If tag starts with I- and matches current, append.
            // Otherwise, flush current.

            bool is_new = (label.rfind("B-", 0) == 0);
            bool is_cont = (label.rfind("I-", 0) == 0);
            std::string pure_tag = (label.size() > 2) ? label.substr(2) : label;
            std::string prev_pure = (current_tag.size() > 2) ? current_tag.substr(2) : current_tag;

            if (is_new || (is_cont && pure_tag != prev_pure) || label == "O") {
                // Flush previous
                if (!current_word.empty()) {
                    entities.push_back({current_word, current_tag, current_score / tokens_in_entity});
                }

                // Reset
                if (label == "O") {
                    current_word = "";
                    current_tag = "O";
                } else {
                    current_word = token_str;
                    current_tag = label; // e.g. B-SKILL
                    current_score = max_val;
                    tokens_in_entity = 1;
                }
            }
            else if (is_cont) {
                // Append
                // Handle WordPiece "##"
                if (token_str.rfind("##", 0) == 0) {
                    current_word += token_str.substr(2);
                } else {
                    // Add space if it's a new word token inside the same entity
                    // (Rough heuristic, better logic involves offset mapping)
                    current_word += " " + token_str;
                }
                current_score += max_val;
                tokens_in_entity++;
            }
        }

        // Flush last
        if (!current_word.empty() && current_tag != "O") {
            entities.push_back({current_word, current_tag, current_score / tokens_in_entity});
        }

        return entities;
    }
};

// =================================================================================
// Public API
// =================================================================================

ResumeParser::ResumeParser(const ParserConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ResumeParser::~ResumeParser() = default;
ResumeParser::ResumeParser(ResumeParser&&) noexcept = default;
ResumeParser& ResumeParser::operator=(ResumeParser&&) noexcept = default;

ParsedResume ResumeParser::parse(const std::string& full_text) {
    if (!pimpl_) throw std::runtime_error("ResumeParser is null.");

    ParsedResume resume;

    // Chunking Loop
    // For simplicity, we split by fixed character length approx equal to tokens
    // Real impl should tokenize first, then stride.

    // Naive Chunking: Split by 1500 chars (~300-400 words)
    int chunk_size = 1500;
    int overlap = 100;

    for (size_t i = 0; i < full_text.length(); i += (chunk_size - overlap)) {
        std::string chunk = full_text.substr(i, chunk_size);

        auto entities = pimpl_->process_chunk(chunk);

        // Aggregate into structured format
        for (const auto& ent : entities) {
            resume.all_entities.push_back(ent);

            // Clean tag (remove B- or I-)
            std::string tag = (ent.label.size() > 2) ? ent.label.substr(2) : ent.label;

            // Mapping based on standard NER datasets (like ResumeDataset)
            if (tag == "Name" && resume.name.empty()) resume.name = ent.text;
            else if (tag == "Email") resume.email = ent.text;
            else if (tag == "Phone") resume.phone = ent.text;
            else if (tag == "Skill" || tag == "Tech") resume.skills.push_back(ent.text);
            else if (tag == "Org" || tag == "Company") resume.experience_orgs.push_back(ent.text);
            else if (tag == "Edu" || tag == "University") resume.education_orgs.push_back(ent.text);
        }
    }

    // Deduplication (Skills often appear multiple times)
    std::sort(resume.skills.begin(), resume.skills.end());
    resume.skills.erase(std::unique(resume.skills.begin(), resume.skills.end()), resume.skills.end());

    return resume;
}

} // namespace xinfer::zoo::recruitment