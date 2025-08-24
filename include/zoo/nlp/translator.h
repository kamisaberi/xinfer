#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace xinfer::zoo::nlp {

    struct TranslatorConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_input_length = 512;
        int max_output_length = 512;
    };

    class Translator {
    public:
        explicit Translator(const TranslatorConfig& config);
        ~Translator();

        Translator(const Translator&) = delete;
        Translator& operator=(const Translator&) = delete;
        Translator(Translator&&) noexcept;
        Translator& operator=(Translator&&) noexcept;

        std::string predict(const std::string& text);

        void predict_stream(const std::string& text,
                            std::function<void(const std::string&)> stream_callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

