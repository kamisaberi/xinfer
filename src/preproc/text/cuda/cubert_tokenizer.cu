#include "cubert_tokenizer.h"
#include <xinfer/core/logging.h>

#include <fstream>
#include <algorithm>
#include <vector>
#include <cstring>

namespace xinfer::preproc::text {

// =================================================================================
// Device Functions
// =================================================================================

// Simple DJB2 Hash (Must match Host implementation)
__device__ __host__ uint32_t hash_djb2(const char* str, int len) {
    uint32_t hash = 5381;
    for (int i = 0; i < len; ++i) {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

struct VocabEntry {
    uint32_t hash;
    int id;
};

// Binary Search on GPU Global Memory
__device__ int lookup_token_id(const VocabEntry* table, int size, uint32_t hash) {
    int left = 0;
    int right = size - 1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (table[mid].hash == hash) return table[mid].id;
        if (table[mid].hash < hash) left = mid + 1;
        else right = mid - 1;
    }
    return -1; // Not found
}

// =================================================================================
// CUDA Kernels
// =================================================================================

/**
 * @brief WordPiece Tokenizer Kernel
 *
 * One thread block per sentence.
 *
 * LIMITATIONS FOR THIS EXAMPLE:
 * - Simplified logic: Assumes words are pre-split by spaces (Basic Tokenization).
 * - Full BERT whitespace/punctuation splitting is complex to do in a single kernel
 *   without shared memory banking conflicts.
 */
__global__ void wordpiece_kernel(
    const char* __restrict__ all_text,
    const int* __restrict__ offsets,
    int* __restrict__ output_ids,
    int* __restrict__ token_counts,
    const VocabEntry* __restrict__ vocab,
    int vocab_size,
    int max_seq_len,
    int num_sequences,
    int unk_id,
    int cls_id,
    int sep_id
) {
    int batch_idx = blockIdx.x;
    if (batch_idx >= num_sequences) return;

    // Get sentence boundaries
    int start = offsets[batch_idx];
    int end = offsets[batch_idx + 1];
    int len = end - start;

    // Output pointer for this sequence
    int* my_output = output_ids + (batch_idx * max_seq_len);

    // Only Thread 0 does the logic (Serial execution per sentence on GPU)
    // To parallelize *within* a sentence requires parallel reduction which is complex for greedy matching.
    // However, since we run thousands of sentences in parallel (Grid size), total throughput is high.
    if (threadIdx.x == 0) {
        int out_idx = 0;

        // Add [CLS]
        my_output[out_idx++] = cls_id;

        int cursor = 0;
        while (cursor < len && out_idx < max_seq_len - 1) {
            // Find next whitespace to delimit "Basic Token"
            // (In a real kernel, we would handle punctuation splitting here)
            int word_end = cursor;
            while (word_end < len && all_text[start + word_end] != ' ') {
                word_end++;
            }

            // WordPiece Greedy Match on sub-string [cursor, word_end]
            int wp_start = cursor;
            bool is_bad = false;

            while (wp_start < word_end && out_idx < max_seq_len - 1) {
                int wp_end = word_end;
                int found_id = -1;

                // Greedy loop: Try longest substring, shrink from right
                while (wp_end > wp_start) {
                    int sub_len = wp_end - wp_start;

                    // TODO: Handle "##" prefix logic for subwords (wp_start > cursor)
                    // For GPU simplicity, we assume hashing handles the ## prefix if we modified the buffer
                    // But here we just lookup raw substring.

                    uint32_t h = hash_djb2(all_text + start + wp_start, sub_len);
                    found_id = lookup_token_id(vocab, vocab_size, h);

                    if (found_id != -1) break;
                    wp_end--;
                }

                if (found_id != -1) {
                    my_output[out_idx++] = found_id;
                    wp_start = wp_end;
                } else {
                    my_output[out_idx++] = unk_id; // UNK
                    wp_start++; // Skip char to avoid inf loop
                }
            }

            cursor = word_end + 1; // Skip space
        }

        // Add [SEP]
        if (out_idx < max_seq_len) my_output[out_idx++] = sep_id;

        // Pad remainder
        while (out_idx < max_seq_len) my_output[out_idx++] = 0; // PAD ID
    }
}

// =================================================================================
// Host Implementation
// =================================================================================

CuBertTokenizer::CuBertTokenizer() {}

CuBertTokenizer::~CuBertTokenizer() {
    if (d_vocab_table) cudaFree(d_vocab_table);
    if (d_text_buffer) cudaFree(d_text_buffer);
    if (d_offsets) cudaFree(d_offsets);
    if (d_output_ids) cudaFree(d_output_ids);
    if (d_token_counts) cudaFree(d_token_counts);
}

void CuBertTokenizer::init(const TokenizerConfig& config) {
    m_config = config;
    load_vocab_to_gpu(config.vocab_path);
}

uint32_t CuBertTokenizer::host_hash_token(const std::string& s) {
    return hash_djb2(s.c_str(), s.length());
}

void CuBertTokenizer::load_vocab_to_gpu(const std::string& path) {
    std::ifstream file(path);
    std::string line;
    std::vector<VocabEntry> host_vocab;
    int id = 0;

    // Read Vocab
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();

        VocabEntry entry;
        entry.hash = host_hash_token(line);
        entry.id = id++;
        host_vocab.push_back(entry);
    }

    // Sort by Hash for Binary Search on GPU
    std::sort(host_vocab.begin(), host_vocab.end(), [](const VocabEntry& a, const VocabEntry& b) {
        return a.hash < b.hash;
    });

    m_vocab_size = host_vocab.size();

    // Upload
    cudaMalloc(&d_vocab_table, m_vocab_size * sizeof(VocabEntry));
    cudaMemcpy(d_vocab_table, host_vocab.data(), m_vocab_size * sizeof(VocabEntry), cudaMemcpyHostToDevice);

    XINFER_LOG_INFO("Uploaded " + std::to_string(m_vocab_size) + " tokens to GPU memory.");
}

void CuBertTokenizer::resize_buffers(size_t total_text_len, size_t batch_size) {
    if (total_text_len > m_text_capacity) {
        if (d_text_buffer) cudaFree(d_text_buffer);
        cudaMalloc(&d_text_buffer, total_text_len);
        m_text_capacity = total_text_len;
    }

    if (batch_size > m_batch_capacity) {
        if (d_offsets) cudaFree(d_offsets);
        if (d_output_ids) cudaFree(d_output_ids);

        cudaMalloc(&d_offsets, (batch_size + 1) * sizeof(int));
        // Output: [Batch, MaxLen]
        cudaMalloc(&d_output_ids, batch_size * m_config.max_length * sizeof(int));

        m_batch_capacity = batch_size;
    }
}

void CuBertTokenizer::process_batch(const std::vector<std::string>& texts,
                                    core::Tensor& input_ids,
                                    core::Tensor& attention_mask) {
    size_t batch_size = texts.size();

    // 1. Flatten strings on Host
    std::vector<char> flat_text;
    std::vector<int> offsets;
    offsets.push_back(0);

    for (const auto& s : texts) {
        // Naive lowercasing (should be done via kernel for speed, but simplifed here)
        // In fully optimized version, text copy is zero-copy pinned memory
        std::string s_lower = s; // Copy
        if (m_config.do_lower_case) {
            std::transform(s_lower.begin(), s_lower.end(), s_lower.begin(), ::tolower);
        }

        flat_text.insert(flat_text.end(), s_lower.begin(), s_lower.end());
        offsets.push_back(flat_text.size());
    }

    // 2. Prepare GPU Memory
    resize_buffers(flat_text.size(), batch_size);

    cudaMemcpy(d_text_buffer, flat_text.data(), flat_text.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(d_offsets, offsets.data(), offsets.size() * sizeof(int), cudaMemcpyHostToDevice);

    // 3. Launch Kernel
    // One block per sequence.
    int threads = 128; // Not used for parallelism within sentence in this naive kernel, but required
    wordpiece_kernel<<<batch_size, threads>>>(
        d_text_buffer,
        d_offsets,
        d_output_ids,
        nullptr, // token_counts unused in this version
        d_vocab_table,
        m_vocab_size,
        m_config.max_length,
        batch_size,
        m_config.unk_token_id,
        m_config.cls_token_id,
        m_config.sep_token_id
    );

    // 4. Copy Output Back to Tensor
    // Resize xInfer Tensors
    input_ids.resize({(int64_t)batch_size, (int64_t)m_config.max_length}, core::DataType::kINT32);
    attention_mask.resize({(int64_t)batch_size, (int64_t)m_config.max_length}, core::DataType::kINT32); // Calc mask later

    // Download Input IDs
    cudaMemcpy(input_ids.data(), d_output_ids, batch_size * m_config.max_length * sizeof(int), cudaMemcpyDeviceToHost);

    // Generate Mask on CPU (Fast enough given data is already here)
    // Or generate it in kernel.
    int32_t* mask_ptr = static_cast<int32_t*>(attention_mask.data());
    int32_t* ids_ptr = static_cast<int32_t*>(input_ids.data());

    for (size_t i = 0; i < batch_size * m_config.max_length; ++i) {
        mask_ptr[i] = (ids_ptr[i] != m_config.pad_token_id) ? 1 : 0;
    }
}

void CuBertTokenizer::process(const std::string& text,
                              core::Tensor& input_ids,
                              core::Tensor& attention_mask) {
    // Wrapper for single item
    std::vector<std::string> batch = {text};
    process_batch(batch, input_ids, attention_mask);
}

std::string CuBertTokenizer::decode(const core::Tensor& output_ids) {
    return ""; // Decoding on GPU not implemented in this snippet
}

} // namespace xinfer::preproc::text