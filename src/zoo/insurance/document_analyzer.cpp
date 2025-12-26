#include <xinfer/zoo/insurance/document_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We are building on top of other Zoo modules! ---
#include <xinfer/zoo/vision/detector.h> // For text detection
#include <xinfer/zoo/vision/ocr.h>      // For text recognition
#include <xinfer/zoo/nlp/question_answering.h>

#include <iostream>
#include <sstream>

namespace xinfer::zoo::insurance {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DocumentAnalyzer::Impl {
    DocAnalyzerConfig config_;

    // --- Components ---
    // Instead of raw engines, we use other Zoo modules

    // Stage 1: OCR
    // OCR itself is often a 2-stage (Detect + Recognize) pipeline
    std::unique_ptr<vision::ObjectDetector> text_detector_;
    std::unique_ptr<vision::OcrRecognizer> text_recognizer_;

    // Stage 2: VQA
    std::unique_ptr<nlp::QuestionAnswering> qa_engine_;

    Impl(const DocAnalyzerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup OCR Pipeline
        // In a real app, you would pass the specific model paths here.
        // For this example, we assume the config contains paths for
        // a text detector and a text recognizer.
        // Let's assume a generic OCR path that contains both for now.

        // This part needs a proper "OCR" zoo module that handles both
        // detection and recognition. We'll use the pieces we have.
        // The OCR model path might be a directory or a single model.
        // Simplified: We are using a generic detector and recognizer.

        vision::DetectorConfig det_cfg;
        // det_cfg.model_path = config_.ocr_model_path + "/detection.onnx";
        text_detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);

        vision::OcrConfig rec_cfg;
        // rec_cfg.model_path = config_.ocr_model_path + "/recognition.onnx";
        text_recognizer_ = std::make_unique<vision::OcrRecognizer>(rec_cfg);

        // 2. Setup VQA Engine
        nlp::QaConfig qa_cfg;
        qa_cfg.model_path = config_.vqa_model_path;
        qa_cfg.vocab_path = config_.vqa_vocab_path;
        qa_cfg.target = config_.target;

        qa_engine_ = std::make_unique<nlp::QuestionAnswering>(qa_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

DocumentAnalyzer::DocumentAnalyzer(const DocAnalyzerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DocumentAnalyzer::~DocumentAnalyzer() = default;
DocumentAnalyzer::DocumentAnalyzer(DocumentAnalyzer&&) noexcept = default;
DocumentAnalyzer& DocumentAnalyzer::operator=(DocumentAnalyzer&&) noexcept = default;

DocumentData DocumentAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DocumentAnalyzer is null.");

    DocumentData result;

    // --- Step 1: Perform OCR ---
    // A. Detect Text Blocks
    auto text_boxes = pimpl_->text_detector_->predict(image);

    // B. Recognize Text in each block
    std::stringstream full_text_ss;
    for (const auto& box : text_boxes) {
        // Crop
        cv::Rect roi((int)box.x1, (int)box.y1, (int)(box.x2-box.x1), (int)(box.y2-box.y1));
        roi &= cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat crop = image(roi);

        auto ocr_res = pimpl_->text_recognizer_->recognize(crop);
        full_text_ss << ocr_res.text << " ";
    }
    result.full_text = full_text_ss.str();

    // --- Step 2: Visual Question Answering ---
    for (const auto& question : pimpl_->config_.queries) {
        // Run VQA engine with the full OCR'd text as context
        auto answer = pimpl_->qa_engine_->ask(question, result.full_text);

        Field field;
        field.question = question;
        field.answer = answer.answer;
        field.confidence = answer.score;
        // To get the bounding box, we'd need to find 'answer.answer'
        // in the original OCR results to get its location, which is a complex
        // text search problem. For now, we leave the box empty.

        result.fields[question] = field;
    }

    return result;
}

} // namespace xinfer::zoo::insurance