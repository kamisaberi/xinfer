#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    /**
     * @brief Pooling Strategy for Sentence Embeddings.
     */
    enum class PoolingType {
        CLS_TOKEN = 0, // Use the vector of the [CLS] token (first token)
        MEAN = 1,      // Average of all visible token vectors
        MAX = 2        // Max-over-time pooling
    };

    struct EmbedderConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., all-MiniLM-L6-v2.onnx)
        std::string model_path;

        // Tokenizer Config
        std::string vocab_path;
        int max_sequence_length = 512;
        bool do_lower_case = true;

        // Embedding Strategy
        PoolingType pooling = PoolingType::MEAN;
        bool normalize = true; // L2 Normalize output (needed for Cosine Similarity)

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Embedder {
    public:
        explicit Embedder(const EmbedderConfig& config);
        ~Embedder();

        // Move semantics
        Embedder(Embedder&&) noexcept;
        Embedder& operator=(Embedder&&) noexcept;
        Embedder(const Embedder&) = delete;
        Embedder& operator=(const Embedder&) = delete;

        /**
         * @brief Generate embedding for a single text.
         *
         * @param text Input string.
         * @return Vector of floats (e.g. 384 dims for MiniLM, 768 for BERT).
         */
        std::vector<float> encode(const std::string& text);

        /**
         * @brief Batch generation.
         *
         * @param texts List of strings.
         * @return List of vectors.
         */
        std::vector<std::vector<float>> encode_batch(const std::vector<std::string>& texts);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp