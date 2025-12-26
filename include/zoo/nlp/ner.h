#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::nlp {

    struct Entity {
        std::string text;    // The extracted phrase (e.g., "Steve Jobs")
        std::string label;   // The category (e.g., "PERSON")
        float confidence;    // Average score of tokens in entity
        int start_char;      // Start index in original string
        int end_char;        // End index
    };

    struct NerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., bert_ner.onnx)
        std::string model_path;

        // Tokenizer Config
        std::string vocab_path;
        int max_sequence_length = 128;
        bool do_lower_case = false; // NER often needs Case Sensitivity

        // Label Map (Path to labels.txt)
        // Format: One label per line, corresponding to model output index.
        // e.g.:
        // 0 O
        // 1 B-PER
        // 2 I-PER
        // ...
        std::string labels_path;

        // Thresholding
        float min_confidence = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class NamedEntityRecognizer {
    public:
        explicit NamedEntityRecognizer(const NerConfig& config);
        ~NamedEntityRecognizer();

        // Move semantics
        NamedEntityRecognizer(NamedEntityRecognizer&&) noexcept;
        NamedEntityRecognizer& operator=(NamedEntityRecognizer&&) noexcept;
        NamedEntityRecognizer(const NamedEntityRecognizer&) = delete;
        NamedEntityRecognizer& operator=(const NamedEntityRecognizer&) = delete;

        /**
         * @brief Extract entities from text.
         *
         * Pipeline:
         * 1. Tokenize (Preserving offsets if possible).
         * 2. Inference (Token Classification).
         * 3. Decode IOB tags (Merge B- and I- tags).
         *
         * @param text Input sentence.
         * @return List of entities found.
         */
        std::vector<Entity> extract(const std::string& text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::nlp