#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct SummarizerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Models ---
        // Seq2Seq models are often split into two parts for inference
        std::string encoder_model_path; // e.g. t5_encoder.engine
        std::string decoder_model_path; // e.g. t5_decoder.engine

        // --- Tokenizer ---
        std::string tokenizer_path; // spiece.model (T5) or vocab.json (BART)
        bool is_t5 = true;          // T5 uses different special tokens than BART

        // --- Generation Params ---
        int max_source_length = 512; // Input limit
        int max_target_length = 128; // Summary limit
        int min_target_length = 10;

        // Sampling
        float temperature = 1.0f; // 1.0 = standard, <1.0 = more focused
        int beam_size = 1;        // 1 = Greedy Search

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Summarizer {
    public:
        explicit Summarizer(const SummarizerConfig& config);
        ~Summarizer();

        // Move semantics
        Summarizer(Summarizer&&) noexcept;
        Summarizer& operator=(Summarizer&&) noexcept;
        Summarizer(const Summarizer&) = delete;
        Summarizer& operator=(const Summarizer&) = delete;

        /**
         * @brief Generate a summary of the input text.
         *
         * Pipeline:
         * 1. Tokenize Input.
         * 2. Run Encoder -> Get Hidden States.
         * 3. Run Decoder Loop (Autoregressive):
         *    - Feed Encoder States + Previous Decoder Tokens.
         *    - Sample next token.
         *    - Repeat until EOS or max length.
         * 4. Detokenize output IDs.
         *
         * @param text Long input document.
         * @return Abstractive summary.
         */
        std::string summarize(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp