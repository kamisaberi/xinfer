#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct TranslatorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Models ---
        // Translation models are often split (Encoder / Decoder)
        std::string encoder_path;
        std::string decoder_path;

        // --- Tokenizer ---
        std::string tokenizer_path; // sentencepiece.model or vocab.json

        // --- Languages ---
        // Default codes (e.g., "eng_Latn", "fra_Latn" for NLLB)
        std::string src_lang;
        std::string tgt_lang;

        // --- Generation ---
        int max_source_length = 128;
        int max_target_length = 128;

        // Beam Search size (1 = Greedy)
        int beam_size = 1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Translator {
    public:
        explicit Translator(const TranslatorConfig& config);
        ~Translator();

        // Move semantics
        Translator(Translator&&) noexcept;
        Translator& operator=(Translator&&) noexcept;
        Translator(const Translator&) = delete;
        Translator& operator=(const Translator&) = delete;

        /**
         * @brief Change translation direction at runtime.
         *
         * @param src Source language code (e.g., "eng_Latn").
         * @param tgt Target language code (e.g., "deu_Latn").
         * @return True if language tokens were found in vocabulary.
         */
        bool set_languages(const std::string& src, const std::string& tgt);

        /**
         * @brief Translate text.
         *
         * Pipeline:
         * 1. Tokenize Input + Add Src Lang Token.
         * 2. Encoder Inference.
         * 3. Decoder Inference (Autoregressive loop starting with Tgt Lang Token).
         * 4. Detokenize.
         *
         * @param text Input string.
         * @return Translated string.
         */
        std::string translate(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp