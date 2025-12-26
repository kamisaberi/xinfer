#include <xinfer/zoo/vision/barcode_scanner.h>
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

struct BarcodeScanner::Impl {
    BarcodeScannerConfig config_;

    // AI Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Classical Decoders (OpenCV)
    cv::QRCodeDetector qr_decoder_;
    // Note: cv::barcode::BarcodeDetector requires OpenCV Contrib.
    // We stick to standard OpenCV QRCodeDetector for basic examples.

    // Tensors
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const BarcodeScannerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Backend
        engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("BarcodeScanner: Failed to load model " + config_.model_path);
        }

        // 2. Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Postprocessor (Object Detection)
        postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        // Barcode models usually have 1 or 2 classes (QR, Barcode)
        // Adjust anchors if model is specialized
        postproc_->init(det_cfg);
    }

    // Helper: Decode a cropped region
    std::pair<std::string, std::string> decode_crop(const cv::Mat& crop) {
        if (crop.empty()) return {"", "Unknown"};

        // Try QR Decode
        std::string data = qr_decoder_.detectAndDecode(crop);
        if (!data.empty()) {
            return {data, "QR_CODE"};
        }

        // Placeholder for 1D Barcode decoding
        // In a production app, you would integrate ZBar or ZXing-C++ here.
        // For standard OpenCV, support is limited without Contrib modules.

        return {"", "Unknown"};
    }
};

// =================================================================================
// Public API
// =================================================================================

BarcodeScanner::BarcodeScanner(const BarcodeScannerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

BarcodeScanner::~BarcodeScanner() = default;
BarcodeScanner::BarcodeScanner(BarcodeScanner&&) noexcept = default;
BarcodeScanner& BarcodeScanner::operator=(BarcodeScanner&&) noexcept = default;

std::vector<BarcodeResult> BarcodeScanner::scan(const cv::Mat& image) {
    if (!pimpl_) return {};

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR; // OpenCV Default

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. AI Detection
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Get Boxes)
    auto detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    std::vector<BarcodeResult> results;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : detections) {
        BarcodeResult res;

        // Scale box back to original image
        res.box.x1 = det.x1 * scale_x;
        res.box.y1 = det.y1 * scale_y;
        res.box.x2 = det.x2 * scale_x;
        res.box.y2 = det.y2 * scale_y;
        res.box.confidence = det.confidence;
        res.decoded = false;

        // 4. Classical Decoding (Optional)
        if (pimpl_->config_.enable_decoding) {
            // Safe Crop
            int x1 = std::max(0, (int)res.box.x1);
            int y1 = std::max(0, (int)res.box.y1);
            int x2 = std::min(image.cols, (int)res.box.x2);
            int y2 = std::min(image.rows, (int)res.box.y2);

            if (x2 > x1 && y2 > y1) {
                cv::Mat crop = image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

                // Enhance crop for better decoding (optional)
                // cv::cvtColor(crop, crop, cv::COLOR_BGR2GRAY);

                auto decoded_info = pimpl_->decode_crop(crop);
                if (!decoded_info.first.empty()) {
                    res.content = decoded_info.first;
                    res.type = decoded_info.second;
                    res.decoded = true;
                }
            }
        }

        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::vision