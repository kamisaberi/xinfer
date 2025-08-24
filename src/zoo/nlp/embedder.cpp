#include <include/zoo/nlp/embedder.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>
#include <numeric>

#include <include/core/engine.h>
// #include <xinfer/preproc/tokenizer.h>

namespace xinfer::zoo::nlp {

struct Embedder::Impl {
    EmbedderConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    // std::unique_ptr<preproc::Tokenizer> tokenizer_;
};

Embedder::Embedder(const EmbedderConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Embedding engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    // pimpl_->tokenizer_ = std::make_unique<preproc::Tokenizer>(pimpl_->config_.vocab_path);
}

Embedder::~Embedder() = default;
Embedder::Embedder(Embedder&&) noexcept = default;
Embedder& Embedder::operator=(Embedder&&) noexcept = default;

TextEmbedding Embedder::predict(const std::string& text) {
    auto batch_result = predict_batch({text});
    return batch_result[0];
}

std::vector<TextEmbedding> Embedder::predict_batch(const std::vector<std::string>& texts) {
    if (!pimpl_) throw std::runtime_error("Embedder is in a moved-from state.");

    // std::vector<std::vector<int64_t>> batch_token_ids;
    // for(const auto& text : texts) {
    //     auto token_ids = pimpl_->tokenizer_->encode(text);
    //     token_ids.resize(pimpl_->config_.max_sequence_length, 0);
    //     batch_token_ids.push_back(token_ids);
    // }

    // int64_t batch_size = texts.size();
    // auto input_shape = pimpl_->engine_->get_input_shape(0);
    // input_shape[0] = batch_size;
    // input_shape[1] = pimpl_->config_.max_sequence_length;

    // core::Tensor input_tensor(input_shape, core::DataType::kINT32);
    // std::vector<int64_t> flat_tokens;
    // for (const auto& tokens : batch_token_ids) {
    //     flat_tokens.insert(flat_tokens.end(), tokens.begin(), tokens.end());
    // }
    // input_tensor.copy_from_host(flat_tokens.data());

    // auto output_tensors = pimpl_->engine_->infer({input_tensor});
    // const core::Tensor& token_embeddings_tensor = output_tensors[0];

    // std::vector<float> h_token_embeddings(token_embeddings_tensor.num_elements());
    // token_embeddings_tensor.copy_to_host(h_token_embeddings.data());

    // auto output_shape = token_embeddings_tensor.shape();
    // int64_t seq_len = output_shape[1];
    // int64_t emb_dim = output_shape[2];

    std::vector<TextEmbedding> final_embeddings;
    // for (int i = 0; i < batch_size; ++i) {
    //     TextEmbedding sentence_embedding(emb_dim, 0.0f);
    //     const float* start_ptr = h_token_embeddings.data() + i * seq_len * emb_dim;

    //     for (int j = 0; j < seq_len; ++j) {
    //         for (int d = 0; d < emb_dim; ++d) {
    //             sentence_embedding[d] += start_ptr[j * emb_dim + d];
    //         }
    //     }

    //     float norm = 0.0f;
    //     for (int d = 0; d < emb_dim; ++d) {
    //         sentence_embedding[d] /= seq_len;
    //         norm += sentence_embedding[d] * sentence_embedding[d];
    //     }
    //     norm = std::sqrt(norm);

    //     if (norm > 1e-6) {
    //         for (float& val : sentence_embedding) {
    //             val /= norm;
    //         }
    //     }
    //     final_embeddings.push_back(sentence_embedding);
    // }

    return final_embeddings;
}

float Embedder::compare(const TextEmbedding& emb1, const TextEmbedding& emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) {
        return 0.0f;
    }
    float dot_product = std::inner_product(emb1.begin(), emb1.end(), emb2.begin(), 0.0f);
    return std::max(0.0f, std::min(1.0f, dot_product));
}

} // namespace xinfer::zoo::nlp