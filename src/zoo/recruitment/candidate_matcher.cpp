#include <xinfer/zoo/recruitment/candidate_matcher.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; embedding math is custom logic below.

#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <set>

namespace xinfer::zoo::recruitment {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CandidateMatcher::Impl {
    MatcherConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::ITextPreprocessor> tokenizer_;

    // Data Containers
    core::Tensor input_ids;
    core::Tensor attention_mask;
    core::Tensor output_hidden_states; // [1, SeqLen, HiddenDim]

    Impl(const MatcherConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CandidateMatcher: Failed to load model " + config_.model_path);
        }

        // 2. Setup Tokenizer
        // Assuming BERT-style WordPiece tokenizer for Sentence-BERT models
        tokenizer_ = preproc::create_text_preprocessor(preproc::text::TokenizerType::BERT_WORDPIECE, config_.target);

        preproc::text::TokenizerConfig tok_cfg;
        tok_cfg.vocab_path = config_.vocab_path;
        tok_cfg.max_length = config_.max_sequence_length;
        tok_cfg.do_lower_case = true;
        tokenizer_->init(tok_cfg);
    }

    // --- Core Logic: Text -> Vector ---
    std::vector<float> get_embedding(const std::string& text) {
        // 1. Tokenize
        tokenizer_->process(text, input_ids, attention_mask);

        // 2. Inference
        // BERT takes {input_ids, attention_mask}
        // Note: Some ONNX exports also require token_type_ids, skipping for simplicity
        engine_->predict({input_ids, attention_mask}, {output_hidden_states});

        // 3. Mean Pooling
        // Convert [1, Seq, Hidden] -> [Hidden]
        auto shape = output_hidden_states.shape();
        int seq_len = (int)shape[1];
        int hidden_dim = (int)shape[2];

        const float* data = static_cast<const float*>(output_hidden_states.data());
        const int32_t* mask = static_cast<const int32_t*>(attention_mask.data()); // Mask is INT32

        std::vector<float> embedding(hidden_dim, 0.0f);
        int valid_tokens = 0;

        for (int i = 0; i < seq_len; ++i) {
            if (mask[i] == 1) { // Only average non-padding tokens
                for (int h = 0; h < hidden_dim; ++h) {
                    embedding[h] += data[i * hidden_dim + h];
                }
                valid_tokens++;
            }
        }

        if (valid_tokens > 0) {
            float inv_count = 1.0f / valid_tokens;
            for (float& v : embedding) v *= inv_count;
        }

        // L2 Normalize
        normalize_vector(embedding);

        return embedding;
    }

    void normalize_vector(std::vector<float>& vec) {
        float sum_sq = 0.0f;
        for (float val : vec) sum_sq += val * val;
        float norm = std::sqrt(sum_sq) + 1e-9f;
        for (float& val : vec) val /= norm;
    }

    float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
        if (a.empty() || b.empty() || a.size() != b.size()) return 0.0f;
        float dot = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
        }
        // Vectors are already L2 normalized, so Dot == Cosine
        return dot;
    }

    float calculate_skill_overlap(const std::vector<std::string>& job_skills,
                                  const std::vector<std::string>& candidate_skills) {
        if (job_skills.empty()) return 1.0f; // No skills required

        std::set<std::string> req_set;
        for(const auto& s : job_skills) {
            // Simple lowercase normalization
            std::string tmp = s;
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            req_set.insert(tmp);
        }

        int matches = 0;
        for(const auto& s : candidate_skills) {
            std::string tmp = s;
            std::transform(tmp.begin(), tmp.end(), tmp.begin(), ::tolower);
            if (req_set.count(tmp)) matches++;
        }

        return (float)matches / req_set.size();
    }
};

// =================================================================================
// Public API
// =================================================================================

CandidateMatcher::CandidateMatcher(const MatcherConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CandidateMatcher::~CandidateMatcher() = default;
CandidateMatcher::CandidateMatcher(CandidateMatcher&&) noexcept = default;
CandidateMatcher& CandidateMatcher::operator=(CandidateMatcher&&) noexcept = default;

std::vector<MatchResult> CandidateMatcher::match(const JobDescription& job,
                                                 const std::vector<CandidateProfile>& candidates) {
    if (!pimpl_) throw std::runtime_error("CandidateMatcher is null.");

    std::vector<MatchResult> results;

    // 1. Embed Job Description
    std::vector<float> job_vec = pimpl_->get_embedding(job.full_text);

    // 2. Score Candidates
    for (const auto& cand : candidates) {
        MatchResult res;
        res.candidate_id = cand.id;

        // A. Semantic Score (Deep Learning)
        std::vector<float> cand_vec = pimpl_->get_embedding(cand.full_text);
        res.semantic_score = pimpl_->cosine_similarity(job_vec, cand_vec);

        // B. Keyword Score (Exact Match)
        res.skill_match_score = pimpl_->calculate_skill_overlap(job.required_skills, cand.skills);

        // C. Weighted Final Score
        res.final_score = (res.semantic_score * pimpl_->config_.weight_semantic) +
                          (res.skill_match_score * pimpl_->config_.weight_skills);

        results.push_back(res);
    }

    // 3. Sort by Final Score Descending
    std::sort(results.begin(), results.end(), [](const MatchResult& a, const MatchResult& b) {
        return a.final_score > b.final_score;
    });

    return results;
}

} // namespace xinfer::zoo::recruitment