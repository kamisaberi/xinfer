#pragma once

#include <xinfer/preproc/text/text_preprocessor.h>
#include <xinfer/preproc/text/types.h>
#include <cuda_runtime.h>
#include <vector>
#include <string>

namespace xinfer::preproc::text {

/**
 * @brief CUDA Accelerated BERT Tokenizer
 *
 * Offloads WordPiece tokenization to the GPU.
 * Optimized for massive batch processing (Blackbox SIEM / Log Analysis).
 *
 * Implementation Strategy:
 * 1. Vocab is hashed and uploaded to GPU global memory (Sorted Hash Table).
 * 2. Input text batch is flattened into a single char buffer and uploaded.
 * 3. CUDA Kernel performs parallel "Greedy Longest-Match-First" lookup.
 */
class CuBertTokenizer : public ITextPreprocessor {
public:
    CuBertTokenizer();
    ~CuBertTokenizer() override;

    void init(const TokenizerConfig& config) override;

    // Single item wrapper (Not recommended for high perf)
    void process(const std::string& text,
                 core::Tensor& input_ids,
                 core::Tensor& attention_mask) override;

    // Batch processing (The optimized path)
    void process_batch(const std::vector<std::string>& texts,
                       core::Tensor& input_ids,
                       core::Tensor& attention_mask);

    std::string decode(const core::Tensor& output_ids) override;

private:
    TokenizerConfig m_config;

    // --- GPU Vocabulary ---
    struct VocabEntry {
        uint32_t hash; // MurmurHash3 of the token string
        int id;        // Token ID
    };

    VocabEntry* d_vocab_table = nullptr; // Sorted array on GPU
    size_t m_vocab_size = 0;

    // --- Dynamic GPU Scratch Buffers ---
    char* d_text_buffer = nullptr;
    int* d_offsets = nullptr;
    int* d_output_ids = nullptr;
    int* d_token_counts = nullptr;

    size_t m_text_capacity = 0;
    size_t m_batch_capacity = 0;

    // --- Helpers ---
    void load_vocab_to_gpu(const std::string& path);
    void resize_buffers(size_t total_text_len, size_t batch_size);
    uint32_t host_hash_token(const std::string& s);
};

} // namespace xinfer::preproc::text