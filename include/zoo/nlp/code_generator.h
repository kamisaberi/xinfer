#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace xinfer::zoo::nlp {

    struct CodeGeneratorConfig {
        std::string engine_path;
        std::string vocab_path;
        int max_new_tokens = 256;
        float temperature = 0.8f;
        float top_p = 0.9f;
    };

    class CodeGenerator {
    public:
        explicit CodeGenerator(const CodeGeneratorConfig& config);
        ~CodeGenerator();

        CodeGenerator(const CodeGenerator&) = delete;
        CodeGenerator& operator=(const CodeGenerator&) = delete;
        CodeGenerator(CodeGenerator&&) noexcept;
        CodeGenerator& operator=(CodeGenerator&&) noexcept;

        std::string predict(const std::string& prompt);

        void predict_stream(const std::string& prompt,
                            std::function<void(const std::string&)> stream_callback);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp

