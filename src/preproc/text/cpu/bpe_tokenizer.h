#pragma once

#include <xinfer/preproc/text/text_preprocessor.h>
#include <map>
#include <vector>

namespace xinfer::preproc::text {

    /**
     * @brief Byte-Pair Encoding (BPE) Tokenizer
     *
     * Used for GPT-2, RoBERTa, Llama.
     * Requires a vocab.json and merges.txt file.
     */
    class BpeTokenizer : public ITextPreprocessor {
    public:
        BpeTokenizer();
        ~BpeTokenizer() override;

        void init(const TokenizerConfig& config) override;
        void process(const std::string& text, core::Tensor& input_ids, core::Tensor& attention_mask) override;
        std::string decode(const core::Tensor& output_ids) override;

    private:
        TokenizerConfig m_config;

        // BPE specific maps
        std::map<std::string, int> m_vocab;
        std::map<std::pair<std::string, std::string>, int> m_bpe_ranks;

        // Helper: Apply merge rules to a word
        std::vector<std::string> bpe(const std::string& token);
    };

}