#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <iomanip>

// xInfer Headers
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/core/tensor.h>

using namespace xinfer;

// -----------------------------------------------------------------------------
// Helper: Compute Cosine Similarity between two vectors
// -----------------------------------------------------------------------------
float cosine_similarity(const std::vector<float>& vec_a, const std::vector<float>& vec_b) {
    if (vec_a.size() != vec_b.size()) return 0.0f;

    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;

    for (size_t i = 0; i < vec_a.size(); ++i) {
        dot += vec_a[i] * vec_b[i];
        norm_a += vec_a[i] * vec_a[i];
        norm_b += vec_b[i] * vec_b[i];
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-9f);
}

// -----------------------------------------------------------------------------
// Helper: Mean Pooling
// Converts [1, SeqLen, Hidden] -> [1, Hidden] by averaging valid tokens
// -----------------------------------------------------------------------------
std::vector<float> mean_pooling(const core::Tensor& last_hidden_state,
                                const core::Tensor& attention_mask) {
    auto shape = last_hidden_state.shape();
    int seq_len = (int)shape[1];
    int hidden_dim = (int)shape[2];

    const float* data = static_cast<const float*>(last_hidden_state.data());
    const int* mask = static_cast<const int*>(attention_mask.data());

    std::vector<float> embedding(hidden_dim, 0.0f);
    int valid_tokens = 0;

    for (int t = 0; t < seq_len; ++t) {
        if (mask[t] == 1) { // Only average non-padding tokens
            valid_tokens++;
            for (int h = 0; h < hidden_dim; ++h) {
                embedding[h] += data[t * hidden_dim + h];
            }
        }
    }

    if (valid_tokens > 0) {
        for (float& val : embedding) val /= valid_tokens;
    }

    return embedding;
}

// -----------------------------------------------------------------------------
// Main Pipeline
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // 1. Configuration
    // Target: CPU is usually fast enough for BERT-base (< 50ms)
    // Use NVIDIA_TRT for batch processing (e.g. indexing documents)
    Target target = Target::INTEL_OV;

    std::string model_path = "bert_base.xml";
    std::string vocab_path = "bert_vocab.txt";

    // 2. Setup Backend
    auto engine = backends::BackendFactory::create(target);
    if (!engine->load_model(model_path)) {
        std::cerr << "Failed to load BERT model." << std::endl;
        return -1;
    }

    // 3. Setup Tokenizer (WordPiece)
    // Uses src/preproc/text/cpu/bert_tokenizer.cpp
    auto tokenizer = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, target);

    preproc::text::TokenizerConfig tok_cfg;
    tok_cfg.vocab_path = vocab_path;
    tok_cfg.max_length = 128; // Sequence length
    tok_cfg.do_lower_case = true;
    tokenizer->init(tok_cfg);

    // 4. Define Inputs
    std::string text_1 = "The bank of the river.";
    std::string text_2 = "The financial bank is open.";

    // We will extract embeddings for both
    std::vector<std::string> inputs_text = {text_1, text_2};
    std::vector<std::vector<float>> embeddings;

    // 5. Inference Loop
    core::Tensor input_ids, attention_mask;
    core::Tensor output_hidden_state; // [1, 128, 768]

    for (const auto& text : inputs_text) {
        std::cout << "Processing: \"" << text << "\"..." << std::endl;

        // A. Tokenize
        tokenizer->process(text, input_ids, attention_mask);

        // B. Inference
        // BERT takes inputs in specific order. Usually [Ids, Mask] or [Ids, Mask, TypeIds]
        // Check your ONNX export!
        engine->predict({input_ids, attention_mask}, {output_hidden_state});

        // C. Pooling (Get Sentence Vector)
        std::vector<float> emb = mean_pooling(output_hidden_state, attention_mask);
        embeddings.push_back(emb);
    }

    // 6. Compare
    float similarity = cosine_similarity(embeddings[0], embeddings[1]);

    std::cout << "\n--------------------------------------------" << std::endl;
    std::cout << "Vector Size: " << embeddings[0].size() << " (Hidden Dim)" << std::endl;
    std::cout << "Cosine Similarity: " << std::fixed << std::setprecision(4) << similarity << std::endl;
    std::cout << "--------------------------------------------" << std::endl;

    if (similarity > 0.8f) std::cout << "-> Semantic Match!" << std::endl;
    else std::cout << "-> Different meanings." << std::endl;

    return 0;
}