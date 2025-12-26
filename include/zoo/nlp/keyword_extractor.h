#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct Keyword {
        std::string phrase;  // The extracted text (e.g., "artificial intelligence")
        float score;         // Confidence
        int start_index;     // Character offset in original text
        int end_index;
    };

    struct KeywordConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., bert_keyword_extractor.onnx)
        // Expected Output: [Batch, SeqLen, NumClasses] (usually 3 classes: O, B, I)
        std::string model_path;

        // Tokenizer Config
        std::string vocab_path;
        int max_sequence_length = 512;
        bool do_lower_case = true;

        // Extraction Settings
        float min_score = 0.5f;   // Minimum confidence to keep a keyword
        bool deduplicate = true;  // Remove duplicate phrases

        // Label Mapping Indices (Model specific)
        int idx_o = 0; // Outside
        int idx_b = 1; // Begin
        int idx_i = 2; // Inside

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class KeywordExtractor {
    public:
        explicit KeywordExtractor(const KeywordConfig& config);
        ~KeywordExtractor();

        // Move semantics
        KeywordExtractor(KeywordExtractor&&) noexcept;
        KeywordExtractor& operator=(KeywordExtractor&&) noexcept;
        KeywordExtractor(const KeywordExtractor&) = delete;
        KeywordExtractor& operator=(const KeywordExtractor&) = delete;

        /**
         * @brief Extract keywords from text.
         *
         * Pipeline:
         * 1. Tokenize (WordPiece).
         * 2. Inference (Token Classification).
         * 3. Grouping (Merge B-tag + I-tags into phrases).
         * 4. Decoding (Token IDs -> String).
         *
         * @param text Input document/paragraph.
         * @return List of extracted keywords.
         */
        std::vector<Keyword> extract(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp