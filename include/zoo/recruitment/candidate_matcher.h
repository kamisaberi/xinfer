#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::recruitment {

    struct CandidateProfile {
        std::string id;
        std::string full_text; // Extracted text from PDF/Docx
        std::vector<std::string> skills; // Explicit skills tags
    };

    struct JobDescription {
        std::string full_text;
        std::vector<std::string> required_skills;
    };

    /**
     * @brief Result of the matching process.
     */
    struct MatchResult {
        std::string candidate_id;

        // Semantic Similarity Score (0.0 - 1.0) based on BERT embeddings
        float semantic_score;

        // Keyword overlap score based on explicit skills
        float skill_match_score;

        // Weighted final score
        float final_score;
    };

    struct MatcherConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // NLP Model Path (e.g., sentence_bert.onnx)
        std::string model_path;

        // Tokenizer Config
        std::string vocab_path;
        int max_sequence_length = 512; // Standard BERT limit

        // Scoring Weights
        float weight_semantic = 0.7f;
        float weight_skills = 0.3f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CandidateMatcher {
    public:
        explicit CandidateMatcher(const MatcherConfig& config);
        ~CandidateMatcher();

        // Move semantics
        CandidateMatcher(CandidateMatcher&&) noexcept;
        CandidateMatcher& operator=(CandidateMatcher&&) noexcept;
        CandidateMatcher(const CandidateMatcher&) = delete;
        CandidateMatcher& operator=(const CandidateMatcher&) = delete;

        /**
         * @brief Match a list of candidates against a job description.
         *
         * Pipeline:
         * 1. Embed Job Description (Sentence-BERT).
         * 2. Embed Candidates.
         * 3. Calculate Cosine Similarity.
         * 4. Calculate Keyword Overlap.
         *
         * @param job The target role.
         * @param candidates List of applicants.
         * @return Ranked list of results.
         */
        std::vector<MatchResult> match(const JobDescription& job,
                                       const std::vector<CandidateProfile>& candidates);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::recruitment