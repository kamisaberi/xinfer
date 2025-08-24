#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace xinfer::zoo::nlp {

    struct SummarizerConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_input_length = 1024;
        int max_summary_length = 150;
    };

    class Summarizer {
    public:
        explicit Summarizer(const SummarizerConfig& config);
        ~Summarizer();

        Summarizer(const Summarizer&) = delete;
        Summarizer& operator=(const Summarizer&) = delete;
        Summarizer(Summarizer&&) noexcept;
        Summarizer& operator=(Summarizer&&) noexcept;

        std::string predict(const std::string& text);

        void predict_stream(const std::string& text,
                            std::function<void(const std::string&)> stream_callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

