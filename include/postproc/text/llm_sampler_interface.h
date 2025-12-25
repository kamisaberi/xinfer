#pragma once
#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::postproc {

    struct LlmSampleConfig {
        float temperature = 1.0f; // Randomness (0.0 = Greedy, 1.0 = Creative)
        int top_k = 50;           // Keep only top K tokens
        float top_p = 0.9f;       // Nucleus sampling (keep top cumsum prob)
        float repetition_penalty = 1.1f;
        int eos_token_id = 2;     // End of Sentence ID
        int vocab_size = 32000;
    };

    class ILlmSampler {
    public:
        virtual ~ILlmSampler() = default;
        virtual void init(const LlmSampleConfig& config) = 0;

        /**
         * @brief Select the next token based on logits.
         *
         * @param logits Raw output [Batch, VocabSize].
         * @param input_ids History of tokens [Batch, SeqLen] (used for rep penalty).
         * @return Selected Token IDs [Batch].
         */
        virtual std::vector<int> sample(const core::Tensor& logits,
                                        const core::Tensor& input_ids) = 0;
    };

}