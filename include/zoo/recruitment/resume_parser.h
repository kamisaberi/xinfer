#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::recruitment {

    /**
     * @brief A specific extracted entity (e.g., "Python" -> SKILL).
     */
    struct ResumeEntity {
        std::string text;
        std::string label; // e.g., "B-SKILL", "I-ORG"
        float confidence;
    };

    /**
     * @brief Structured Resume Data.
     */
    struct ParsedResume {
        std::string name;
        std::string email;
        std::string phone;
        std::vector<std::string> skills;
        std::vector<std::string> experience_orgs; // Companies worked at
        std::vector<std::string> education_orgs;  // Universities

        // Raw list of all detected entities
        std::vector<ResumeEntity> all_entities;
    };

    struct ParserConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., bert_ner_resume.onnx)
        // Should be a Token Classification model (Output: [Batch, Seq, NumLabels])
        std::string model_path;

        // Tokenizer Config
        std::string vocab_path;

        // Label Map (Path to tags.txt: "O", "B-SKILL", "I-SKILL"...)
        std::string labels_path;

        // Context Window
        int max_seq_length = 512;
        int chunk_overlap = 50; // Overlap to prevent cutting entities in half

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ResumeParser {
    public:
        explicit ResumeParser(const ParserConfig& config);
        ~ResumeParser();

        // Move semantics
        ResumeParser(ResumeParser&&) noexcept;
        ResumeParser& operator=(ResumeParser&&) noexcept;
        ResumeParser(const ResumeParser&) = delete;
        ResumeParser& operator=(const ResumeParser&) = delete;

        /**
         * @brief Parse resume text.
         *
         * Pipeline:
         * 1. Tokenize (Sliding Window / Chunking).
         * 2. Inference (Batch BERT).
         * 3. Decode Tags (ArgMax).
         * 4. Entity Aggregation (Merge B-tag and I-tag tokens).
         *
         * @param full_text Raw text extracted from PDF.
         * @return Structured data.
         */
        ParsedResume parse(const std::string& full_text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::recruitment