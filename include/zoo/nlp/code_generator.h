#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct CodeGenConfig {
        // Hardware Target (High-end GPU is best for LLMs)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., codellama_7b_int4.engine)
        std::string model_path;

        // Tokenizer Config
        std::string tokenizer_path; // model.model or tokenizer.json
        bool is_bpe = true;         // True for Llama/GPT, False for BERT

        // Generation Parameters
        int max_new_tokens = 128;
        int context_window = 2048;  // Total sequence length limit

        float temperature = 0.2f;   // Low temp for code precision
        float top_p = 0.95f;
        float repetition_penalty = 1.1f;

        // Stop tokens (e.g., "<EOT>", "\nclass")
        // Tokenizer usually handles IDs, but high-level strings are useful here
        std::vector<std::string> stop_words;

        // Vendor flags (e.g. "USE_KV_CACHE=1")
        std::vector<std::string> vendor_params;
    };

    /**
     * @brief Callback for streaming tokens as they are generated.
     * Return true to continue generation, false to stop.
     */
    using StreamCallback = std::function<bool(const std::string& token)>;

    class CodeGenerator {
    public:
        explicit CodeGenerator(const CodeGenConfig& config);
        ~CodeGenerator();

        // Move semantics
        CodeGenerator(CodeGenerator&&) noexcept;
        CodeGenerator& operator=(CodeGenerator&&) noexcept;
        CodeGenerator(const CodeGenerator&) = delete;
        CodeGenerator& operator=(const CodeGenerator&) = delete;

        /**
         * @brief Generate code based on a prompt.
         *
         * @param prompt The code context (e.g., "void sort_vector(std::vector<int>& v) {").
         * @return The completed code string.
         */
        std::string generate(const std::string& prompt);

        /**
         * @brief Generate code with streaming output.
         * Useful for IDE plugins to show text as it appears.
         */
        void generate_stream(const std::string& prompt, StreamCallback callback);

        /**
         * @brief Clear KV Cache (Context).
         * Call this when starting a completely new file/task.
         */
        void reset_context();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp