#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::nlp {

    struct QAResult {
        std::string answer;
        float score;
        int start_pos;
        int end_pos;
    };

    struct QAConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_sequence_length = 384;
    };

    class QuestionAnswering {
    public:
        explicit QuestionAnswering(const QAConfig& config);
        ~QuestionAnswering();

        QuestionAnswering(const QuestionAnswering&) = delete;
        QuestionAnswering& operator=(const QuestionAnswering&) = delete;
        QuestionAnswering(QuestionAnswering&&) noexcept;
        QuestionAnswering& operator=(QuestionAnswering&&) noexcept;

        QAResult predict(const std::string& question, const std::string& context);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

