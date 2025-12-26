#include <xinfer/zoo/vision/license_plate_recognizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FaceDetector::Impl { // This should be LPRImpl for clarity
    FaceDetectorConfig config_; // Should be LPRConfig

    // Detection Components
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;

    // OCR Components
    std::unique_ptr<backends::IBackend> ocr_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> ocr_preproc_;
    std::unique_ptr<postproc::IOcrPostprocessor> ocr_postproc_;

    // Data Containers
    core::Tensor det_input, det_output;
    core::Tensor ocr_input, ocr_output;

    Impl(const FaceDetectorConfig& config) : config_(config) { // LPRConfig
        initialize();
    }

    void initialize() {
        // --- Initialize Detector ---
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config det_backend_cfg;
        det_backend_cfg.model_path = config_.model_path;
        det_engine_->load_model(det_backend_cfg.model_path);

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig det_pre_cfg;
        det_pre_cfg.target_width = config_.input_width;
        det_pre_cfg.target_height = config_.input_height;
        det_preproc_->init(det_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_post_cfg;
        det_post_cfg.conf_threshold = config_.conf_threshold;
        det_post_cfg.nms_threshold = config_.nms_threshold;
        det_postproc_->init(det_cfg);

        // --- Initialize OCR ---
        ocr_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config ocr_backend_cfg;
        ocr_backend_cfg.model_path = config_.ocr_model_path;
        ocr_engine_->load_model(ocr_backend_cfg.model_path);

        ocr_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig ocr_pre_cfg;
        ocr_pre_cfg.target_width = config_.ocr_input_width;
        ocr_pre_cfg.target_height = config_.ocr_input_height;
        ocr_pre_cfg.target_format = preproc::ImageFormat::RGB;
        ocr_preproc_->init(ocr_pre_cfg);

        ocr_postproc_ = postproc::create_ocr(config_.target);
        postproc::OcrConfig ocr_cfg;
        ocr_cfg.blank_index = config_.blank_index;
        ocr_cfg.vocabulary = config_.vocabulary; // Assuming vocab loaded elsewhere or passed as string
        ocr_postproc_->init(ocr_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

LicensePlateRecognizer::LicensePlateRecognizer(const FaceDetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(static_cast<const FaceDetectorConfig&>(config))) {} // Cast needed due to copy

LicensePlateRecognizer::~LicenseRecognizer() = default;
LicensePlateRecognizer::LicensePlateRecognizer(LicensePlateRecognizer&&) noexcept = default;
LicensePlateRecognizer& LicenseRecognizer::operator=(LicenseRecognizer&&) noexcept = default;

std::vector<LicensePlateResult> LicensePlateRecognizer::scan(const cv::Mat& image) {
    std::vector<LicensePlateResult> results;
    if (!pimpl_) return results;

    // --- Step 1: Detect Plate ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    auto det_boxes = pimpl_->det_postproc_->process({pimpl_->output_tensor});

    // --- Step 2: Crop & Recognize Each Plate ---
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : det_boxes) {
        // Crop detected plate region
        cv::Rect det_rect(det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1);
        // Scale detected box back to original image size
        cv::Rect scaled_det_rect(
            static_cast<int>(det.x1 * scale_x),
            static_cast<int>(det.y1 * scale_y),
            static_cast<int>((det.x2 - det.x1) * scale_x),
            static_cast<int>((det.y2 - det.y1) * scale_y)
        );

        // Ensure crop is within image bounds
        cv::Rect final_roi = scaled_det_rect & cv::Rect(0, 0, image.cols, image.rows);

        if (final_roi.width > 0 && final_roi.height > 0) {
            cv::Mat plate_crop = image(final_roi);

            // Preprocess crop for OCR Model
            preproc::ImageFrame plate_frame;
            plate_frame.data = plate_crop.data;
            plate_frame.width = plate_crop.cols;
            plate_frame.height = plate_crop.rows;
            plate_frame.format = preproc::ImageFormat::BGR;

            pimpl_->ocr_preproc_->process(plate_frame, pimpl_->ocr_input);

            // OCR Inference
            pimpl_->ocr_engine_->predict({pimpl_->ocr_input}, {pimpl_->ocr_output});

            // Decode Text
            // Note: OCR decoder returns a vector of strings (if batching was used)
            std::vector<std::string> decoded_texts = pimpl_->ocr_postproc_->process(pimpl_->ocr_output);

            LicensePlateResult result;
            if (!decoded_texts.empty()) {
                result.content = decoded_texts[0]; // Assuming batch size 1
                result.decoded = true;
            } else {
                result.content = "";
                result.decoded = false;
            }
            result.box = { (float)scaled_det_rect.x, (float)scaled_det_rect.y,
                           (float)(scaled_det_rect.x + scaled_det_rect.width),
                           (float)(scaled_det_rect.y + scaled_det_rect.height),
                           det.confidence, det.class_id };
            results.push_back(result);
        }
    }

    return results;
}

} // namespace xinfer::zoo::vision