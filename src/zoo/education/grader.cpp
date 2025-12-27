#include <xinfer/zoo/education/grader.h>
#include <xinfer/core/logging.h>

// --- We reuse other Zoo Modules ---
#include <xinfer/zoo/nlp/embedder.h>
#include <xinfer/zoo/vision/ocr.h>

#include <iostream>
#include <cmath>
#include <algorithm>

namespace xinfer::zoo::education {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Grader::Impl {
    GraderConfig config_;

    // --- Components ---
    std::unique_ptr<nlp::Embedder> embedder_;
    std::unique_ptr<vision::OcrRecognizer> ocr_;

    // --- Caches ---
    // Cache embeddings of model answers to avoid re-computing
    std::map<std::string, std::vector<float>> rubric_cache_;

    Impl(const GraderConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Semantic Similarity Embedder
        nlp::EmbedderConfig embed_cfg;
        embed_cfg.target = config_.target;
        embed_cfg.model_path = config_.similarity_model_path;
        embed_cfg.vocab_path = config_.similarity_vocab_path;
        embed_cfg.pooling = nlp::PoolingType::MEAN;
        embed_cfg.normalize = true; // For Cosine Similarity

        embedder_ = std::make_unique<nlp::Embedder>(embed_cfg);

        // 2. Setup OCR (if provided)
        if (!config_.ocr_model_path.empty()) {
            vision::OcrConfig ocr_cfg;
            ocr_cfg.target = config_.target;
            ocr_cfg.model_path = config_.ocr_model_path;
            // The vocab for OCR is usually just characters
            ocr_cfg.vocabulary = "0123456789abcdefghijklmnopqrstuvwxyz";
            ocr_cfg.blank_index = 0;

            ocr_ = std::make_unique<vision::OcrRecognizer>(ocr_cfg);
        }
    }

    // --- Core Logic ---

    // Get embedding for a model answer (with caching)
    const std::vector<float>& get_rubric_embedding(const RubricItem& rubric) {
        if (rubric_cache_.find(rubric.question_id) == rubric_cache_.end()) {
            rubric_cache_[rubric.question_id] = embedder_->encode(rubric.model_answer);
        }
        return rubric_cache_.at(rubric.question_id);
    }
};

// =================================================================================
// Public API
// =================================================================================

Grader::Grader(const GraderConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Grader::~Grader() = default;
Grader::Grader(Grader&&) noexcept = default;
Grader& Grader::operator=(Grader&&) noexcept = default;

GradeResult Grader::grade_text(const RubricItem& rubric, const std::string& student_answer) {
    if (!pimpl_ || !pimpl_->embedder_) throw std::runtime_error("Grader is not initialized.");

    GradeResult result;
    result.question_id = rubric.question_id;

    if (rubric.type == QuestionType::MULTIPLE_CHOICE) {
        // Simple string comparison
        // Normalize: " a." -> "a"
        std::string clean_student = student_answer;
        clean_student.erase(std::remove_if(clean_student.begin(), clean_student.end(), ::isspace), clean_student.end());
        std::transform(clean_student.begin(), clean_student.end(), clean_student.begin(), ::tolower);

        result.is_correct = (clean_student == rubric.model_answer);
        result.score = result.is_correct ? 1.0f : 0.0f;
        result.justification = result.is_correct ? "Correct" : "Incorrect";
    }
    else if (rubric.type == QuestionType::SHORT_ANSWER_TEXT) {
        // Semantic Comparison

        // 1. Get Embeddings
        const auto& rubric_vec = pimpl_->get_rubric_embedding(rubric);
        auto student_vec = pimpl_->embedder_->encode(student_answer);

        // 2. Compute Similarity
        // We can reuse the static helper from FaceRecognizer or ImageSimilarity
        // For now, re-implementing dot product:
        float dot = 0.0f;
        for(size_t i=0; i<rubric_vec.size(); ++i) dot += rubric_vec[i] * student_vec[i];

        result.score = dot;
        result.is_correct = (result.score >= pimpl_->config_.correctness_threshold);

        if (result.is_correct) result.justification = "Correct";
        else if (result.score > 0.4f) result.justification = "Partial Match";
        else result.justification = "Incorrect";
    }

    return result;
}

GradeResult Grader::grade_image(const RubricItem& rubric, const cv::Mat& student_image) {
    if (!pimpl_ || !pimpl_->ocr_) {
        XINFER_LOG_ERROR("OCR model not configured for handwritten grading.");
        return {rubric.question_id, false, 0.0f, "OCR Error"};
    }

    // 1. Transcribe
    auto ocr_res = pimpl_->ocr_->recognize(student_image);

    // 2. Grade the transcribed text
    return grade_text(rubric, ocr_res.text);
}

} // namespace xinfer::zoo::education