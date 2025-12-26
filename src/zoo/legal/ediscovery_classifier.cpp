#include <xinfer/zoo/legal/ediscovery_classifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; multi-label sigmoid logic is custom.

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>

namespace xinfer::zoo::legal {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct EDiscoveryClassifier::Impl {
    EDiscoveryConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_logits; // [Batch, NumLabels]

    // Labels
    std::vector<std::string> labels_;

    Impl(const EDiscoveryConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("eDiscovery: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = true;
        tokenizer_->init(tok_cfg);

        // 3. Load Labels
        load_labels(config_.labels_path);
    }

    void load_labels(const std::string& path) {
        if (path.empty()) return;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }

    // --- Post-processing for Multi-Label ---
    std::vector<DocumentTag> decode_multilabel(const float* logits) {
        std::vector<DocumentTag> tags;

        for (size_t i = 0; i < labels_.size(); ++i) {
            float logit = logits[i];

            // Apply Sigmoid: 1 / (1 + exp(-x))
            float prob = 1.0f / (1.0f + std::exp(-logit));

            if (prob > config_.classification_threshold) {
                tags.push_back({labels_[i], prob});
            }
        }
        return tags;
    }
};

// =================================================================================
// Public API
// =================================================================================

EDiscoveryClassifier::EDiscoveryClassifier(const EDiscoveryConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

EDiscoveryClassifier::~EDiscoveryClassifier() = default;
EDiscoveryClassifier::EDiscoveryClassifier(EDiscoveryClassifier&&) noexcept = default;
EDiscoveryClassifier& EDiscoveryClassifier::operator=(EDiscoveryClassifier&&) noexcept = default;

EDiscoveryResult EDiscoveryClassifier::classify(const std::string& doc_id, const std::string& doc_text) {
    auto batch_res = classify_batch({{doc_id, doc_text}});
    return batch_res.empty() ? EDiscoveryResult{doc_id, {}, false} : batch_res[0];
}

std::vector<EDiscoveryResult> EDiscoveryClassifier::classify_batch(const std::map<std::string, std::string>& documents) {
    if (!pimpl_) throw std::runtime_error("eDiscovery is null.");

    size_t batch_size = documents.size();
    if (batch_size == 0) return {};

    // 1. Tokenize Batch
    // A proper batch tokenizer in preproc would be faster.
    // For now, we manually build the batch tensor.

    size_t seq_len = pimpl_->config_.max_sequence_length;
    pimpl_->input_ids.resize({(int64_t)batch_size, (int64_t)seq_len}, core::DataType::kINT32);
    pimpl_->attention_mask.resize({(int64_t)batch_size, (int64_t)seq_len}, core::DataType::kINT32);

    int32_t* ids_ptr = static_cast<int32_t*>(pimpl_->input_ids.data());
    int32_t* mask_ptr = static_cast<int32_t*>(pimpl_->attention_mask.data());

    int row_idx = 0;
    std::vector<std::string> doc_ids; // Preserve order
    doc_ids.reserve(batch_size);

    for (const auto& kv : documents) {
        doc_ids.push_back(kv.first);

        core::Tensor row_ids, row_mask;
        pimpl_->tokenizer_->process(kv.second, row_ids, row_mask);

        // Copy to batch
        std::memcpy(ids_ptr + row_idx * seq_len, row_ids.data(), seq_len * sizeof(int32_t));
        std::memcpy(mask_ptr + row_idx * seq_len, row_mask.data(), seq_len * sizeof(int32_t));
        row_idx++;
    }

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_ids, pimpl_->attention_mask}, {pimpl_->output_logits});

    // 3. Postprocess
    std::vector<EDiscoveryResult> results;
    results.reserve(batch_size);
    const float* logits_ptr = static_cast<const float*>(pimpl_->output_logits.data());
    int num_labels = pimpl_->labels_.size();

    for (size_t i = 0; i < batch_size; ++i) {
        EDiscoveryResult res;
        res.document_id = doc_ids[i];

        // Point to start of logits for this document
        const float* doc_logits = logits_ptr + (i * num_labels);

        res.tags = pimpl_->decode_multilabel(doc_logits);

        // Check for specific "Relevant" tag (or similar)
        res.is_responsive = false;
        for (const auto& tag : res.tags) {
            if (tag.tag == "Relevant" || tag.tag == "Responsive") {
                res.is_responsive = true;
                break;
            }
        }

        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::legal