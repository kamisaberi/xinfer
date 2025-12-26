#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::insurance {

    /**
     * @brief A single piece of extracted information.
     */
    struct Field {
        std::string question; // The query, e.g., "Policy Number"
        std::string answer;
        float confidence;
        postproc::BoundingBox box; // Where the answer was found
    };

    /**
     * @brief Structured data from the document.
     */
    struct DocumentData {
        // Map from Field Name -> Value
        std::map<std::string, Field> fields;

        // Full OCR text for reference
        std::string full_text;
    };

    struct DocAnalyzerConfig {
        // Hardware Target (LLMs/LayoutLMs usually require GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: OCR ---
        // (e.g., DBNet for detection + CRNN for recognition, or a single paddleocr model)
        // For simplicity, we assume a single OCR engine that does both
        std::string ocr_model_path;
        int ocr_input_width = 960;
        int ocr_input_height = 960;

        // --- Model 2: VQA/LayoutLM ---
        // Takes text + layout info to answer questions
        std::string vqa_model_path;
        std::string vqa_vocab_path;

        // --- Logic ---
        // List of questions to ask the document
        std::vector<std::string> queries = {"Policy Number", "Claim Amount", "Date of Incident"};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DocumentAnalyzer {
    public:
        explicit DocumentAnalyzer(const DocAnalyzerConfig& config);
        ~DocumentAnalyzer();

        // Move semantics
        DocumentAnalyzer(DocumentAnalyzer&&) noexcept;
        DocumentAnalyzer& operator=(DocumentAnalyzer&&) noexcept;
        DocumentAnalyzer(const DocumentAnalyzer&) = delete;
        DocumentAnalyzer& operator=(const DocumentAnalyzer&) = delete;

        /**
         * @brief Analyze a document image.
         *
         * @param image Input image of a form/document.
         * @return Structured data.
         */
        DocumentData analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::insurance