#pragma once

#include <xinfer/preproc/text/text_preprocessor.h>
#include <map>
#include <vector>
#include <set>
#include <string>

namespace xinfer::preproc::text {

    class BpeTokenizer : public ITextPreprocessor {
    public:
        BpeTokenizer();
        ~BpeTokenizer() override;

        void init(const TokenizerConfig& config) override;

        void process(const std::string& text,
                     core::Tensor& input_ids,
                     core::Tensor& attention_mask) override;

        std::string decode(const core::Tensor& output_ids) override;

    private:
        TokenizerConfig m_config;

        // Maps token string -> ID
        std::map<std::string, int> m_vocab;
        // Maps ID -> token string
        std::map<int, std::string> m_decoder;

        // Maps pair(token_a, token_b) -> Rank (lower is better priority)
        std::map<std::pair<std::string, std::string>, int> m_bpe_ranks;

        // Cache for BPE results (word -> list of tokens)
        std::map<std::string, std::vector<std::string>> m_cache;

        // Byte Encoder map (Byte -> Unicode Char)
        std::map<unsigned char, std::string> m_byte_encoder;
        std::map<std::string, unsigned char> m_byte_decoder;

        // Helpers
        void load_vocab(const std::string& path);
        void load_merges(const std::string& path);
        void build_byte_encoder();

        std::vector<std::string> bpe(const std::string& token);
        std::vector<std::string> get_pairs(const std::vector<std::string>& word);
    };

} // namespace xinfer::preproc::text