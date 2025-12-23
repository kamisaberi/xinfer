#pragma once

#include <xinfer/preproc/text/text_preprocessor.h>
#include <xinfer/preproc/text/types.h>
#include <map>
#include <vector>
#include <string>

namespace xinfer::preproc::text {

    /**
     * @brief BERT WordPiece Tokenizer (CPU)
     *
     * Implements the standard BERT tokenization pipeline:
     * 1. Basic Tokenization (Punctuation splitting, Lowercasing)
     * 2. WordPiece Tokenization (Greedy longest-match-first)
     * 3. ID Conversion & Padding
     */
    class BertTokenizer : public ITextPreprocessor {
    public:
        BertTokenizer();
        ~BertTokenizer() override;

        void init(const TokenizerConfig& config) override;

        /**
         * @brief Tokenize text into Input IDs and Attention Mask.
         */
        void process(const std::string& text,
                     core::Tensor& input_ids,
                     core::Tensor& attention_mask) override;

        /**
         * @brief Convert IDs back to string (Detokenization).
         */
        std::string decode(const core::Tensor& output_ids) override;

    private:
        TokenizerConfig m_config;

        // Vocabulary Map (Token string -> Int ID)
        std::map<std::string, int> m_vocab;

        // Inverse Vocabulary (Int ID -> Token string) for decoding
        std::map<int, std::string> m_ids_to_tokens;

        // --- Helper Methods ---
        void load_vocab(const std::string& path);

        // Performs "Basic Tokenization" (Whitespace + Punctuation split)
        std::vector<std::string> basic_tokenize(const std::string& text);

        // Performs "WordPiece" algorithm (splitting unknown words into subwords like "play", "##ing")
        std::vector<int> wordpiece_tokenize(const std::string& word);
    };

} // namespace xinfer::preproc::text