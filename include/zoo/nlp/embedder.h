#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::nlp {

using TextEmbedding = std::vector<float>;

struct EmbedderConfig {
    std::string engine_path;
    std::string vocab_path;
    int max_sequence_length = 256;
};

class Embedder {
public:
    explicit Embedder(const EmbedderConfig& config);
    ~Embedder();

    Embedder(const Embedder&) = delete;
    Embedder& operator=(const Embedder&) = delete;
    Embedder(Embedder&&) noexcept;
    Embedder& operator=(Embedder&&) noexcept;

    TextEmbedding predict(const std::string& text);
    std::vector<TextEmbedding> predict_batch(const std::vector<std::string>& texts);

    static float compare(const TextEmbedding& emb1, const TextEmbedding& emb2);

private:
    struct Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace xinfer::zoo::nlp

