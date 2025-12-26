#include <xinfer/zoo/nlp/question_answering.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; QA span selection logic is specific.

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xinfer::zoo::nlp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct QuestionAnswering::Impl {
    QaConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor token_type_ids; // [1, SeqLen] (Segment IDs)

    // Outputs
    core::Tensor output_start_logits;
    core::Tensor output_end_logits;

    Impl(const QaConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("QA: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = config_.do_lower_case;
        // We handle special tokens manually for QA pairing
        tok_cfg.add_special_tokens = false;
        tokenizer_->init(tok_cfg);

        // 3. Pre-allocate Segment ID Tensor
        token_type_ids.resize({1, (int64_t)config_.max_sequence_length}, core::DataType::kINT32);
    }

    // --- Helper: Construct Input Sequence ---
    void prepare_inputs(const std::string& question, const std::string& context) {
        // Tokenize separately to get lengths
        // Note: Using temporary tensors, in real optimization use resize on cached buffers
        core::Tensor q_ids, q_mask, c_ids, c_mask;

        tokenizer_->process(question, q_ids, q_mask);
        tokenizer_->process(context, c_ids, c_mask);

        // Raw pointers
        const int32_t* q_ptr = static_cast<const int32_t*>(q_ids.data());
        const int32_t* c_ptr = static_cast<const int32_t*>(c_ids.data());

        // Count actual tokens (ignoring padding 0)
        int q_len = 0; while(q_len < q_ids.shape()[1] && q_ptr[q_len] != 0) q_len++;
        int c_len = 0; while(c_len < c_ids.shape()[1] && c_ptr[c_len] != 0) c_len++;

        // Prepare Destination Buffers
        input_ids.resize({1, (int64_t)config_.max_sequence_length}, core::DataType::kINT32);
        attention_mask.resize({1, (int64_t)config_.max_sequence_length}, core::DataType::kINT32);
        token_type_ids.resize({1, (int64_t)config_.max_sequence_length}, core::DataType::kINT32);

        int32_t* ids_dst = static_cast<int32_t*>(input_ids.data());
        int32_t* mask_dst = static_cast<int32_t*>(attention_mask.data());
        int32_t* type_dst = static_cast<int32_t*>(token_type_ids.data());

        // Fill Sequence: [CLS] Q... [SEP] C... [SEP]
        int idx = 0;
        int max_len = config_.max_sequence_length;

        // [CLS]
        ids_dst[idx] = 101; mask_dst[idx] = 1; type_dst[idx] = 0; idx++;

        // Question
        for(int i=0; i<q_len && idx < max_len - 2; ++i) { // Reserve space for SEPs
            ids_dst[idx] = q_ptr[i]; mask_dst[idx] = 1; type_dst[idx] = 0; idx++;
        }

        // [SEP]
        ids_dst[idx] = 102; mask_dst[idx] = 1; type_dst[idx] = 0; idx++;

        // Context
        for(int i=0; i<c_len && idx < max_len - 1; ++i) { // Reserve space for last SEP
            ids_dst[idx] = c_ptr[i]; mask_dst[idx] = 1; type_dst[idx] = 1; idx++;
        }

        // [SEP]
        ids_dst[idx] = 102; mask_dst[idx] = 1; type_dst[idx] = 1; idx++;

        // Padding
        while(idx < max_len) {
            ids_dst[idx] = 0; mask_dst[idx] = 0; type_dst[idx] = 0; idx++;
        }
    }

    // --- Post-Processing: Find Best Span ---
    QaResult decode_output() {
        QaResult result;
        result.score = -1e9;

        // Pointers
        const float* start_logits = static_cast<const float*>(output_start_logits.data());
        const float* end_logits = static_cast<const float*>(output_end_logits.data());

        int seq_len = config_.max_sequence_length;

        // Search for best (start, end) pair
        // Constraint: start <= end
        // Constraint: end - start < max_answer_len
        // Constraint: start/end are within the Context segment (Token Type ID = 1)

        const int32_t* type_ids = static_cast<const int32_t*>(token_type_ids.data());

        int best_start = 0;
        int best_end = 0;

        for (int i = 0; i < seq_len; ++i) {
            // Optimization: Skip if not in context segment (ignores Q and special tokens)
            // (Note: Some models allow answer in Question, but usually it's Context)
            // if (type_ids[i] != 1) continue;

            for (int j = i; j < std::min(i + config_.max_answer_len, seq_len); ++j) {
                // if (type_ids[j] != 1) continue;

                float score = start_logits[i] + end_logits[j];
                if (score > result.score) {
                    result.score = score;
                    best_start = i;
                    best_end = j;
                }
            }
        }

        result.start_token_idx = best_start;
        result.end_token_idx = best_end;

        // Extract Text
        // Create a temporary tensor containing the span of IDs
        int span_len = best_end - best_start + 1;
        core::Tensor span_tensor({1, (int64_t)span_len}, core::DataType::kINT32);
        int32_t* span_ptr = static_cast<int32_t*>(span_tensor.data());

        const int32_t* all_ids = static_cast<const int32_t*>(input_ids.data());
        std::memcpy(span_ptr, all_ids + best_start, span_len * sizeof(int32_t));

        result.answer = tokenizer_->decode(span_tensor);
        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

QuestionAnswering::QuestionAnswering(const QaConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

QuestionAnswering::~QuestionAnswering() = default;
QuestionAnswering::QuestionAnswering(QuestionAnswering&&) noexcept = default;
QuestionAnswering& QuestionAnswering::operator=(QuestionAnswering&&) noexcept = default;

QaResult QuestionAnswering::ask(const std::string& question, const std::string& context) {
    if (!pimpl_) throw std::runtime_error("QuestionAnswering is null.");

    // 1. Prepare (Tokenize & Construct Pair)
    pimpl_->prepare_inputs(question, context);

    // 2. Inference
    // BERT QA models usually take 3 inputs: Ids, Mask, SegmentIds
    pimpl_->engine_->predict(
        {pimpl_->input_ids, pimpl_->attention_mask, pimpl_->token_type_ids},
        {pimpl_->output_start_logits, pimpl_->output_end_logits}
    );

    // 3. Postprocess
    return pimpl_->decode_output();
}

} // namespace xinfer::zoo::nlp