#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct QaResult {
        std::string answer;  // The extracted text
        float score;         // Confidence (Logit sum)
        int start_token_idx;
        int end_token_idx;
    };

    struct QaConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., bert_squad.onnx)
        // Expected Outputs: [1, SeqLen] (Start Logits), [1, SeqLen] (End Logits)
        std::string model_path;

        // Tokenizer
        std::string vocab_path;
        int max_sequence_length = 384; // QA often uses 384 or 512
        bool do_lower_case = true;

        // Logic
        int max_answer_len = 30; // Ignore answers longer than this (likely garbage)

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class QuestionAnswering {
    public:
        explicit QuestionAnswering(const QaConfig& config);
        ~QuestionAnswering();

        // Move semantics
        QuestionAnswering(QuestionAnswering&&) noexcept;
        QuestionAnswering& operator=(QuestionAnswering&&) noexcept;
        QuestionAnswering(const QuestionAnswering&) = delete;
        QuestionAnswering& operator=(const QuestionAnswering&) = delete;

        /**
         * @brief Find the answer to a question within a context.
         *
         * Pipeline:
         * 1. Tokenize: `[CLS] Question [SEP] Context [SEP]`
         * 2. Inference: Get Start/End logits.
         * 3. Decode: Find best valid span (i, j) where i <= j.
         *
         * @param question The query (e.g. "Who wrote this?")
         * @param context The paragraph to search in.
         * @return The best answer found.
         */
        QaResult ask(const std::string& question, const std::string& context);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp