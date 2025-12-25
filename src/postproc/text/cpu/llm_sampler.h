#pragma once

#include <xinfer/postproc/text/llm_sampler_interface.h>
#include <random>
#include <vector>

namespace xinfer::postproc {

    /**
     * @brief CPU Implementation of LLM Token Sampling.
     *
     * Supports:
     * - Temperature Scaling
     * - Top-K Sampling
     * - Top-P (Nucleus) Sampling
     * - Repetition Penalty
     * - Greedy Decoding (Temperature = 0)
     *
     * This is efficient enough for CPU execution because the vocabulary size (32k-128k)
     * is small enough for modern CPUs to sort/filter in microseconds.
     */
    class CpuLlmSampler : public ILlmSampler {
    public:
        CpuLlmSampler();
        ~CpuLlmSampler() override;

        void init(const LlmSampleConfig& config) override;

        /**
         * @brief Selects the next token.
         *
         * @param logits Raw output from LLM [Batch, VocabSize].
         * @param input_ids History of tokens [Batch, SeqLen] (used for repetition penalty).
         * @return Vector of selected token IDs (one per batch item).
         */
        std::vector<int> sample(const core::Tensor& logits,
                                const core::Tensor& input_ids) override;

    private:
        LlmSampleConfig m_config;
        std::mt19937 m_rng; // Mersenne Twister for high-quality randomness

        // Helper to apply softmax in-place
        void apply_softmax(std::vector<float>& scores);
    };

} // namespace xinfer::postproc