#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::education {

    enum class QuestionType {
        SHORT_ANSWER_TEXT = 0,
        HANDWRITTEN_IMAGE = 1,
        MULTIPLE_CHOICE = 2
    };

    /**
     * @brief A single question and its model answer.
     */
    struct RubricItem {
        std::string question_id;
        QuestionType type;
        std::string model_answer; // Text for SHORT_ANSWER, or "A", "B" for MULTIPLE_CHOICE
    };

    /**
     * @brief Result of grading.
     */
    struct GradeResult {
        std::string question_id;
        bool is_correct;
        float score; // 0.0 to 1.0 (Similarity score for text)
        std::string justification; // "Correct", "Incorrect", "Partial Match"
    };

    struct GraderConfig {
        // Hardware Target (CPU is fine for single student, GPU for batch)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Semantic Similarity (Sentence-BERT) ---
        std::string similarity_model_path;
        std::string similarity_vocab_path;

        // --- Model 2: Handwriting OCR (Optional) ---
        std::string ocr_model_path;
        std::string ocr_vocab_path;

        // --- Logic ---
        float correctness_threshold = 0.75f; // Cosine similarity > 0.75 is a match

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Grader {
    public:
        explicit Grader(const GraderConfig& config);
        ~Grader();

        // Move semantics
        Grader(Grader&&) noexcept;
        Grader& operator=(Grader&&) noexcept;
        Grader(const Grader&) = delete;
        Grader& operator=(const Grader&) = delete;

        /**
         * @brief Grade a student's answer.
         *
         * @param rubric The question and correct answer.
         * @param student_answer The student's submission (text or image).
         * @return The grade.
         */
        GradeResult grade_text(const RubricItem& rubric, const std::string& student_answer);
        GradeResult grade_image(const RubricItem& rubric, const cv::Mat& student_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::education