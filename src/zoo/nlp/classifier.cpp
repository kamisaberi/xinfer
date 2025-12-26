#include <xinfer/zoo/nlp/classifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TextClassifier::Impl {
    TextClassifierConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_logits;

    // Labels
    std::vector<std::string> labels_;

    Impl(const TextClassifierConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TextClassifier: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer (Preproc)
        // Defaulting to BERT WordPiece for standard classifiers.
        // Could be made configurable if using RoBERTa (BPE).
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = config_.do_lower_case;
        tokenizer_->init(tok_cfg);

        // 3. Setup Post-processor
        // We reuse the generic classification logic (Softmax + TopK)
        postproc_ = postproc::create_classification(config_.target);

        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = config_.top_k;
        post_cfg.apply_softmax = true; // NLP models usually output raw logits

        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
            post_cfg.labels = labels_;
        }

        postproc_->init(post_cfg);
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            XINFER_LOG_WARN("Could not open labels file: " + path);
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

TextClassifier::TextClassifier(const TextClassifierConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TextClassifier::~TextClassifier() = default;
TextClassifier::TextClassifier(TextClassifier&&) noexcept = default;
TextClassifier& TextClassifier::operator=(TextClassifier&&) noexcept = default;

std::vector<TextClassResult> TextClassifier::classify(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("TextClassifier is null.");

    // 1. Tokenize
    pimpl_->tokenizer_->process(text, pimpl_->input_ids, pimpl_->attention_mask);

    // 2. Inference
    // Standard BERT input: {input_ids, attention_mask}
    // Some models may require token_type_ids, but for single sentence classification
    // it's usually 0 and often optional in exported ONNX.
    pimpl_->engine_->predict({pimpl_->input_ids, pimpl_->attention_mask}, {pimpl_->output_logits});

    // 3. Postprocess
    auto raw_results = pimpl_->postproc_->process(pimpl_->output_logits);

    std::vector<TextClassResult> results;
    if (!raw_results.empty()) {
        const auto& batch_res = raw_results[0]; // Batch size 1
        results.reserve(batch_res.size());

        for (const auto& item : batch_res) {
            TextClassResult res;
            res.id = item.id;
            res.confidence = item.score;
            res.label = item.label;

            // Fallback label generation if file wasn't provided
            if (res.label.empty()) {
                if (item.id < (int)pimpl_->labels_.size()) {
                    res.label = pimpl_->labels_[item.id];
                } else {
                    res.label = "Class_" + std::to_string(item.id);
                }
            }
            results.push_back(res);
        }
    }

    return results;
}

} // namespace xinfer::zoo::nlp