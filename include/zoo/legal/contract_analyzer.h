#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::legal {

    /**
     * @brief An extracted entity from the text.
     */
    struct LegalEntity {
        std::string text;
        std::string type; // "PARTY", "DATE", "JURISDICTION"
        float confidence;
        int start_char, end_char;
    };

    /**
     * @brief An analyzed paragraph or clause.
     */
    struct AnalyzedClause {
        std::string text;
        std::string clause_type; // "Termination", "Liability", etc.
        float type_confidence;
        std::vector<LegalEntity> entities;
    };

    struct ContractResult {
        std::vector<AnalyzedClause> clauses;
        // High-level summary of extracted key terms
        std::vector<std::string> all_parties;
        std::vector<std::string> all_dates;
    };

    struct AnalyzerConfig {
        // Hardware Target (CPU/Intel OpenVINO is fine for document processing)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Clause Classifier ---
        // (e.g., bert_legal_classifier.onnx)
        std::string clause_model_path;
        std::string clause_labels_path;

        // --- Model 2: NER Extractor ---
        // (e.g., bert_legal_ner.onnx)
        std::string ner_model_path;
        std::string ner_labels_path;

        // --- Tokenizer (Shared by both models) ---
        std::string vocab_path;
        int max_seq_length = 512;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ContractAnalyzer {
    public:
        explicit ContractAnalyzer(const ContractConfig& config);
        ~ContractAnalyzer();

        // Move semantics
        ContractAnalyzer(ContractAnalyzer&&) noexcept;
        ContractAnalyzer& operator=(ContractAnalyzer&&) noexcept;
        ContractAnalyzer(const ContractAnalyzer&) = delete;
        ContractAnalyzer& operator=(const ContractAnalyzer&) = delete;

        /**
         * @brief Analyze a full legal contract text.
         *
         * @param full_text The text of the contract.
         * @return Structured analysis.
         */
        ContractResult analyze(const std::string& full_text);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::legal