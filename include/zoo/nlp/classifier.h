#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    /**
     * @brief Result of text classification.
     */
    struct TextClassResult {
        int id;             // Class Index
        float confidence;   // Probability (0.0 - 1.0)
        std::string label;  // Class Name (e.g., "Positive", "Spam")
    };

    struct TextClassifierConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., distilbert_sentiment.onnx)
        std::string model_path;

        // Tokenizer settings
        std::string vocab_path;    // vocab.txt or tokenizer.json
        int max_sequence_length = 128;
        bool do_lower_case = true;

        // Label Map (Path to text file with labels, one per line)
        std::string labels_path;

        // Post-processing
        int top_k = 1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TextClassifier {
    public:
        explicit TextClassifier(const TextClassifierConfig& config);
        ~TextClassifier();

        // Move semantics
        TextClassifier(TextClassifier&&) noexcept;
        TextClassifier& operator=(TextClassifier&&) noexcept;
        TextClassifier(const TextClassifier&) = delete;
        TextClassifier& operator=(const TextClassifier&) = delete;

        /**
         * @brief Classify a text string.
         *
         * Pipeline:
         * 1. Tokenize (WordPiece/BPE) -> Input IDs + Mask.
         * 2. Inference (Transformer).
         * 3. Postprocess (Softmax + TopK).
         *
         * @param text Input string.
         * @return List of top_k predicted classes.
         */
        std::vector<TextClassResult> classify(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp