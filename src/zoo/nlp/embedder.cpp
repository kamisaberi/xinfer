#include <xinfer/zoo/nlp/embedder.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; embedding pooling is custom math.

#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Embedder::Impl {
    EmbedderConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_states; // [Batch, SeqLen, Hidden]

    Impl(const EmbedderConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Embedder: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        // Usually BERT-WordPiece
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = config_.do_lower_case;
        tokenizer_->init(tok_cfg);
    }

    // --- Math: Pooling & Normalization ---
    std::vector<float> pool_and_normalize(const core::Tensor& hidden_states,
                                          const core::Tensor& mask,
                                          int batch_idx) {
        auto shape = hidden_states.shape();
        int seq_len = (int)shape[1];
        int hidden_dim = (int)shape[2];

        const float* data = static_cast<const float*>(hidden_states.data()) + (batch_idx * seq_len * hidden_dim);
        const int32_t* mask_ptr = static_cast<const int32_t*>(mask.data()) + (batch_idx * seq_len);

        std::vector<float> embedding(hidden_dim, 0.0f);

        if (config_.pooling == PoolingType::CLS_TOKEN) {
            // [CLS] is usually at index 0
            for (int h = 0; h < hidden_dim; ++h) {
                embedding[h] = data[0 * hidden_dim + h];
            }
        }
        else if (config_.pooling == PoolingType::MEAN) {
            int valid_tokens = 0;
            for (int t = 0; t < seq_len; ++t) {
                if (mask_ptr[t] == 1) { // Only valid tokens
                    for (int h = 0; h < hidden_dim; ++h) {
                        embedding[h] += data[t * hidden_dim + h];
                    }
                    valid_tokens++;
                }
            }
            if (valid_tokens > 0) {
                float scale = 1.0f / valid_tokens;
                for (float& val : embedding) val *= scale;
            }
        }
        else if (config_.pooling == PoolingType::MAX) {
            std::fill(embedding.begin(), embedding.end(), -1e9f);
            for (int t = 0; t < seq_len; ++t) {
                if (mask_ptr[t] == 1) {
                    for (int h = 0; h < hidden_dim; ++h) {
                        embedding[h] = std::max(embedding[h], data[t * hidden_dim + h]);
                    }
                }
            }
        }

        // L2 Normalization
        if (config_.normalize) {
            float sum_sq = 0.0f;
            for (float val : embedding) sum_sq += val * val;
            float norm = std::sqrt(sum_sq) + 1e-9f;
            float inv_norm = 1.0f / norm;
            for (float& val : embedding) val *= inv_norm;
        }

        return embedding;
    }
};

// =================================================================================
// Public API
// =================================================================================

Embedder::Embedder(const EmbedderConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Embedder::~Embedder() = default;
Embedder::Embedder(Embedder&&) noexcept = default;
Embedder& Embedder::operator=(Embedder&&) noexcept = default;

std::vector<float> Embedder::encode(const std::string& text) {
    if (!pimpl_) throw std::runtime_error("Embedder is null.");
    auto batch_res = encode_batch({text});
    if (batch_res.empty()) return {};
    return batch_res[0];
}

std::vector<std::vector<float>> Embedder::encode_batch(const std::vector<std::string>& texts) {
    if (!pimpl_) throw std::runtime_error("Embedder is null.");

    // Note: This simple implementation tokenizes one-by-one.
    // Ideally, the Preprocessor should support batch_process() to fill tensor rows in parallel.
    // For now, we reuse single process logic loop.

    // 1. Resize Tensors for Batch
    size_t batch_size = texts.size();
    size_t seq_len = pimpl_->config_.max_sequence_length;

    pimpl_->input_ids.resize({(int64_t)batch_size, (int64_t)seq_len}, core::DataType::kINT32);
    pimpl_->attention_mask.resize({(int64_t)batch_size, (int64_t)seq_len}, core::DataType::kINT32);

    // 2. Tokenize Loop
    // To do this properly with the current interface, we need to manually offset into the tensor buffer.
    // This assumes ITextPreprocessor::process writes to a [1, SeqLen] tensor.
    // We create temporary tensors for single rows and copy.

    core::Tensor row_ids, row_mask;
    int32_t* batch_ids_ptr = static_cast<int32_t*>(pimpl_->input_ids.data());
    int32_t* batch_mask_ptr = static_cast<int32_t*>(pimpl_->attention_mask.data());

    for (size_t i = 0; i < batch_size; ++i) {
        pimpl_->tokenizer_->process(texts[i], row_ids, row_mask);

        // Copy to batch tensor
        std::memcpy(batch_ids_ptr + (i * seq_len), row_ids.data(), seq_len * sizeof(int32_t));
        std::memcpy(batch_mask_ptr + (i * seq_len), row_mask.data(), seq_len * sizeof(int32_t));
    }

    // 3. Inference
    pimpl_->engine_->predict({pimpl_->input_ids, pimpl_->attention_mask}, {pimpl_->output_states});

    // 4. Pooling
    std::vector<std::vector<float>> results;
    results.reserve(batch_size);

    for (size_t i = 0; i < batch_size; ++i) {
        results.push_back(pimpl_->pool_and_normalize(pimpl_->output_states, pimpl_->attention_mask, i));
    }

    return results;
}

} // namespace xinfer::zoo::nlp