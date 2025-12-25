#include "llm_sampler.h"
#include <xinfer/core/logging.h>

#include <cmath>
#include <algorithm>
#include <numeric>
#include <set>
#include <iostream>

namespace xinfer::postproc {

CpuLlmSampler::CpuLlmSampler() {
    // Seed with random device
    std::random_device rd;
    m_rng = std::mt19937(rd());
}

CpuLlmSampler::~CpuLlmSampler() {}

void CpuLlmSampler::init(const LlmSampleConfig& config) {
    m_config = config;
}

void CpuLlmSampler::apply_softmax(std::vector<float>& scores) {
    float max_val = *std::max_element(scores.begin(), scores.end());
    float sum = 0.0f;

    for (float& s : scores) {
        s = std::exp(s - max_val); // Stable softmax
        sum += s;
    }

    float inv_sum = 1.0f / sum;
    for (float& s : scores) {
        s *= inv_sum;
    }
}

std::vector<int> CpuLlmSampler::sample(const core::Tensor& logits, const core::Tensor& input_ids) {
    std::vector<int> next_tokens;

    // 1. Shapes & Pointers
    auto logit_shape = logits.shape();
    int batch_size = (int)logit_shape[0];
    int vocab_size = (int)logit_shape[1]; // e.g. 32000

    const float* logits_ptr = static_cast<const float*>(logits.data());

    // History handling (for repetition penalty)
    const int* history_ptr = nullptr;
    int history_len = 0;
    if (input_ids.size() > 0) {
        history_ptr = static_cast<const int*>(input_ids.data());
        history_len = (int)input_ids.shape()[1];
    }

    // Process each item in the batch
    for (int b = 0; b < batch_size; ++b) {
        // Copy logits to a local vector for modification
        // Offset: b * vocab_size
        const float* row_start = logits_ptr + (b * vocab_size);
        std::vector<float> scores(row_start, row_start + vocab_size);

        // --- Step A: Repetition Penalty ---
        if (m_config.repetition_penalty != 1.0f && history_ptr != nullptr) {
            // Collect seen tokens for this batch index
            const int* batch_hist = history_ptr + (b * history_len);
            std::set<int> seen_tokens(batch_hist, batch_hist + history_len);

            for (int token_id : seen_tokens) {
                if (token_id >= 0 && token_id < vocab_size) {
                    float& score = scores[token_id];
                    // If score < 0, multiply to push it further down
                    // If score > 0, divide to push it down
                    if (score < 0) score *= m_config.repetition_penalty;
                    else score /= m_config.repetition_penalty;
                }
            }
        }

        // --- Step B: Temperature Scaling ---
        if (m_config.temperature > 0.0f) {
            float inv_temp = 1.0f / m_config.temperature;
            for (float& s : scores) s *= inv_temp;
        }

        // --- Step C: Softmax ---
        // Converts logits (-inf to +inf) into probabilities (0.0 to 1.0)
        apply_softmax(scores);

        // --- Step D: Sampling Strategy ---

        // Pair of (probability, index) for sorting
        std::vector<std::pair<float, int>> probs_indices(vocab_size);
        for (int i = 0; i < vocab_size; ++i) {
            probs_indices[i] = {scores[i], i};
        }

        // Strategy 1: Greedy (Temperature ~ 0)
        if (m_config.temperature < 1e-5f) {
            auto best = std::max_element(probs_indices.begin(), probs_indices.end());
            next_tokens.push_back(best->second);
            continue;
        }

        // Strategy 2: Top-K Filtering
        // Keep only the K most likely tokens
        if (m_config.top_k > 0 && m_config.top_k < vocab_size) {
            std::partial_sort(probs_indices.begin(),
                              probs_indices.begin() + m_config.top_k,
                              probs_indices.end(),
                              [](const auto& a, const auto& b) { return a.first > b.first; });

            // Resize to keep only Top K
            probs_indices.resize(m_config.top_k);
        }

        // Strategy 3: Top-P (Nucleus) Filtering
        // Keep tokens whose cumulative probability >= P
        if (m_config.top_p < 1.0f && m_config.top_p > 0.0f) {
            // Sort descending (if not already sorted by Top-K)
            if (m_config.top_k <= 0) {
                std::sort(probs_indices.begin(), probs_indices.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
            }

            float cum_prob = 0.0f;
            std::vector<std::pair<float, int>> nucleus;

            for (const auto& p : probs_indices) {
                nucleus.push_back(p);
                cum_prob += p.first;
                if (cum_prob >= m_config.top_p) break;
            }
            probs_indices = nucleus;
        }

        // --- Step E: Random Selection ---
        // Select from the filtered list based on probabilities
        std::vector<float> final_probs;
        final_probs.reserve(probs_indices.size());
        for (const auto& p : probs_indices) final_probs.push_back(p.first);

        std::discrete_distribution<> dist(final_probs.begin(), final_probs.end());
        int sampled_index = dist(m_rng);

        next_tokens.push_back(probs_indices[sampled_index].second);
    }

    return next_tokens;
}

} // namespace xinfer::postproc