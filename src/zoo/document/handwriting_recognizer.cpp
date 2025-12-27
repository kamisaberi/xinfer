#include <xinfer/zoo/document/handwriting_recognizer.h>
#include <xinfer/core/logging.h>

// --- Reuse the generic OCR module ---
#include <xinfer/zoo/vision/ocr.h>

#include <iostream>

namespace xinfer::zoo::document {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct HandwritingRecognizer::Impl {
    HwrConfig config_;

    // We use the generic OcrRecognizer as the engine
    std::unique_ptr<vision::OcrRecognizer> ocr_engine_;

    Impl(const HwrConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the underlying OCR engine with our specific settings
        vision::OcrConfig ocr_cfg;
        ocr_cfg.target = config_.target;
        ocr_cfg.model_path = config_.model_path;

        // Handwriting models are often trained on grayscale [-1, 1]
        ocr_cfg.input_width = config_.input_width;
        ocr_cfg.input_height = config_.input_height;
        ocr_cfg.mean = {127.5f};
        ocr_cfg.std = {127.5f};

        ocr_cfg.vocabulary = config_.vocab_path; // Pass path or raw string
        ocr_cfg.blank_index = config_.blank_index;

        ocr_cfg.vendor_params = config_.vendor_params;

        ocr_engine_ = std::make_unique<vision::OcrRecognizer>(ocr_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

HandwritingRecognizer::HandwritingRecognizer(const HwrConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

HandwritingRecognizer::~HandwritingRecognizer() = default;
HandwritingRecognizer::HandwritingRecognizer(HandwritingRecognizer&&) noexcept = default;
HandwritingRecognizer& HandwritingRecognizer::operator=(HandwritingRecognizer&&) noexcept = default;

HwrResult HandwritingRecognizer::recognize(const cv::Mat& image_line) {
    if (!pimpl_ || !pimpl_->ocr_engine_) {
        throw std::runtime_error("HandwritingRecognizer is not initialized.");
    }

    // 1. Pre-processing specific to handwriting (optional)
    // For example, binarization or skew correction
    cv::Mat processed_image = image_line;
    // cv::threshold(image_line, processed_image, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 2. Delegate to the generic OCR engine
    auto ocr_res = pimpl_->ocr_engine_->recognize(processed_image);

    // 3. Map to HWR result format
    HwrResult result;
    result.text = ocr_res.text;
    result.confidence = ocr_res.confidence;

    return result;
}

} // namespace xinfer::zoo::document