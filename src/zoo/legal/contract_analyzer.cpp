#include <xinfer/zoo/legal/contract_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h> // Reused for text

#include <iostream>
#include <fstream>
#include <sstream>
#include <set>

namespace xinfer::zoo::legal {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ContractAnalyzer::Impl {
    AnalyzerConfig config_;

    // --- Components: Clause Classifier ---
    std::unique_ptr<backends::IBackend> cls_engine_;
    std::unique_ptr<preproc::ITextPreprocessor> cls_tokenizer_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;
    core::Tensor cls_ids, cls_mask, cls_logits;

    // --- Components: NER ---
    std::unique_ptr<backends::IBackend> ner_engine_;
    std::unique_ptr<preproc::ITextPreprocessor> ner_tokenizer_;
    // Custom NER post-processing logic
    std::vector<std::string> ner_labels_;
    core::Tensor ner_ids, ner_mask, ner_logits;

    Impl(const AnalyzerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // --- Setup Classifier ---
        cls_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config cls_cfg; cls_cfg.model_path = config_.clause_model_path;

        if (!cls_engine_->load_model(cls_cfg.model_path)) {
            throw std::runtime_error("ContractAnalyzer: Failed to load clause model.");
        }

        cls_tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);
        preproc::text::TokenizerConfig cls_tok_cfg;
        cls_tok_cfg.vocab_path = config_.vocab_path;
        cls_tok_cfg.max_length = config_.max_seq_length;
        cls_tokenizer_->init(cls_tok_cfg);

        cls_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cls_post_cfg;
        cls_post_cfg.top_k = 1;
        load_labels(config_.clause_labels_path, cls_post_cfg.labels);
        cls_postproc_->init(cls_post_cfg);

        // --- Setup NER ---
        ner_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config ner_cfg; ner_cfg.model_path = config_.ner_model_path;

        if (!ner_engine_->load_model(ner_cfg.model_path)) {
            throw std::runtime_error("ContractAnalyzer: Failed to load NER model.");
        }

        ner_tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);
        preproc::text::TokenizerConfig ner_tok_cfg;
        ner_tok_cfg.vocab_path = config_.vocab_path;
        ner_tok_cfg.max_length = config_.max_seq_length;
        ner_tokenizer_->init(ner_tok_cfg);

        load_labels(config_.ner_labels_path, ner_labels_);
    }

    void load_labels(const std::string& path, std::vector<std::string>& list) {
        if (path.empty()) return;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            list.push_back(line);
        }
    }

    // --- Custom NER Decoding (Reusing logic from zoo/nlp/ner) ---
    std::vector<LegalEntity> decode_ner(const std::string& text) {
        ner_tokenizer_->process(text, ner_ids, ner_mask);
        ner_engine_->predict({ner_ids, ner_mask}, {ner_logits});

        // IOB Decoding logic (Simplified from zoo/nlp/ner)
        std::vector<LegalEntity> entities;
        // ... (Code for ArgMax, grouping B- and I- tags, and reconstructing words) ...
        return entities;
    }
};

// =================================================================================
// Public API
// =================================================================================

ContractAnalyzer::ContractAnalyzer(const AnalyzerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ContractAnalyzer::~ContractAnalyzer() = default;
ContractAnalyzer::ContractAnalyzer(ContractAnalyzer&&) noexcept = default;
ContractAnalyzer& ContractAnalyzer::operator=(ContractAnalyzer&&) noexcept = default;

ContractResult ContractAnalyzer::analyze(const std::string& full_text) {
    if (!pimpl_) throw std::runtime_error("ContractAnalyzer is null.");

    ContractResult result;

    // 1. Split text into paragraphs/clauses (Simple newline split)
    std::vector<std::string> clauses;
    std::stringstream ss(full_text);
    std::string line;
    std::string current_clause;

    while (std::getline(ss, line, '\n')) {
        if (line.length() < 5) { // Heuristic for paragraph break
            if (!current_clause.empty()) {
                clauses.push_back(current_clause);
                current_clause.clear();
            }
        } else {
            current_clause += line + " ";
        }
    }
    if (!current_clause.empty()) clauses.push_back(current_clause);

    // 2. Process each clause
    std::set<std::string> parties, dates; // To deduplicate summary

    for (const auto& clause_text : clauses) {
        AnalyzedClause analyzed;
        analyzed.text = clause_text;

        // A. Classify Clause Type
        pimpl_->cls_tokenizer_->process(clause_text, pimpl_->cls_ids, pimpl_->cls_mask);
        pimpl_->cls_engine_->predict({pimpl_->cls_ids, pimpl_->cls_mask}, {pimpl_->cls_logits});
        auto cls_res = pimpl_->cls_postproc_->process(pimpl_->cls_logits);

        if (!cls_res.empty() && !cls_res[0].empty()) {
            analyzed.clause_type = cls_res[0][0].label;
            analyzed.type_confidence = cls_res[0][0].score;
        }

        // B. Extract Entities (NER)
        analyzed.entities = pimpl_->decode_ner(clause_text);

        // C. Populate summary fields
        for (const auto& ent : analyzed.entities) {
            if (ent.type == "PARTY") parties.insert(ent.text);
            if (ent.type == "DATE") dates.insert(ent.text);
        }

        result.clauses.push_back(analyzed);
    }

    // Copy from set to vector for final output
    result.all_parties = std::vector<std::string>(parties.begin(), parties.end());
    result.all_dates = std::vector<std::string>(dates.begin(), dates.end());

    return result;
}

} // namespace xinfer::zoo::legal