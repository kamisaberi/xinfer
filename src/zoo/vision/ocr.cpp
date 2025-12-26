#include <xinfer/zoo/vision/ocr.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <numeric>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct OcrRecognizer::Impl {
    OcrConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IOcrPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const OcrConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("OcrRecognizer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        // Most OCR models are trained on Grayscale, but some take RGB.
        // We assume RGB here; user config should handle channel mapping via mean/std if needed.
        // Or we could expose ImageFormat in config.
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = config_.scale_factor;

        preproc_->init(pre_cfg);

        // 3. Setup Post-processor (CTC Decoder)
        postproc_ = postproc::create_ocr(config_.target);

        postproc::OcrConfig ocr_cfg;
        ocr_cfg.vocabulary = config_.vocabulary;
        ocr_cfg.blank_index = config_.blank_index;
        ocr_cfg.min_confidence = 0.0f; // Return everything, let app decide

        postproc_->init(ocr_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

OcrRecognizer::OcrRecognizer(const OcrConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

OcrRecognizer::~OcrRecognizer() = default;
OcrRecognizer::OcrRecognizer(OcrRecognizer&&) noexcept = default;
OcrRecognizer& OcrRecognizer::operator=(OcrRecognizer&&) noexcept = default;

OcrResult OcrRecognizer::recognize(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("OcrRecognizer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    // Assume input is BGR (OpenCV default)
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (CTC Decode)
    // The generic OCR post-processor returns a vector of strings (for batching)
    std::vector<std::string> texts = pimpl_->postproc_->process(pimpl_->output_tensor);

    OcrResult result;
    if (!texts.empty()) {
        result.text = texts[0];
        // TODO: Implement confidence calculation in IOcrPostprocessor interface
        // to return scores alongside strings. For now, placeholder 1.0.
        result.confidence = 1.0f;
    } else {
        result.text = "";
        result.confidence = 0.0f;
    }

    return result;
}

} // namespace xinfer::zoo::vision