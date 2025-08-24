#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace xinfer::zoo::nlp {

    struct TextGeneratorConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_new_tokens = 512;
        float temperature = 0.7f;
        float top_p = 0.9f;
    };

    class TextGenerator {
    public:
        explicit TextGenerator(const TextGeneratorConfig& config);
        ~TextGenerator();

        TextGenerator(const TextGenerator&) = delete;
        TextGenerator& operator=(const TextGenerator&) = delete;
        TextGenerator(TextGenerator&&) noexcept;
        TextGenerator& operator=(TextGenerator&&) noexcept;

        std::string predict(const std::string& prompt);

        void predict_stream(const std::string& prompt,
                            std::function<void(const std::string&)> stream_callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

