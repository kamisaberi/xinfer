#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::nlp {

    struct Keyword {
        std::string text;
        float score;
    };

    struct KeywordExtractorConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_sequence_length = 512;
    };

    class KeywordExtractor {
    public:
        explicit KeywordExtractor(const KeywordExtractorConfig& config);
        ~KeywordExtractor();

        KeywordExtractor(const KeywordExtractor&) = delete;
        KeywordExtractor& operator=(const KeywordExtractor&) = delete;
        KeywordExtractor(KeywordExtractor&&) noexcept;
        KeywordExtractor& operator=(KeywordExtractor&&) noexcept;

        std::vector<Keyword> predict(const std::string& text, int top_k = 5);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

