#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct TextGenConfig {
        // Hardware Target (NVIDIA_TRT is recommended for LLMs)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., llama3_8b.engine, gpt2.onnx)
        std::string model_path;

        // Tokenizer Config
        std::string tokenizer_path; // vocab.json, tokenizer.model
        bool is_llama = true;       // Affects special token handling (BOS/EOS)

        // Generation Hyperparameters
        int max_new_tokens = 128;
        int context_window = 4096;

        float temperature = 0.7f;   // Creative writing
        float top_p = 0.9f;         // Nucleus sampling
        int top_k = 50;
        float repetition_penalty = 1.2f;

        // System Prompt (Optional prepend)
        std::string system_prompt;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    /**
     * @brief Callback for streaming text generation.
     * @param token The newly generated text chunk.
     * @return true to continue, false to abort generation.
     */
    using TextStreamCallback = std::function<bool(const std::string& token)>;

    class TextGenerator {
    public:
        explicit TextGenerator(const TextGenConfig& config);
        ~TextGenerator();

        // Move semantics
        TextGenerator(TextGenerator&&) noexcept;
        TextGenerator& operator=(TextGenerator&&) noexcept;
        TextGenerator(const TextGenerator&) = delete;
        TextGenerator& operator=(const TextGenerator&) = delete;

        /**
         * @brief Generate text based on a prompt.
         * Blocks until completion.
         *
         * @param prompt User input.
         * @return Full generated response.
         */
        std::string generate(const std::string& prompt);

        /**
         * @brief Generate text with streaming output.
         * Useful for chat interfaces.
         */
        void generate_stream(const std::string& prompt, TextStreamCallback callback);

        /**
         * @brief Clear conversation history / KV cache.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp