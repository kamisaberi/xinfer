#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    struct BarcodeResult {
        std::string content; // Decoded text (e.g., URL, Product ID)
        std::string type;    // "QR_CODE", "EAN-13", or "Unknown"
        postproc::BoundingBox box; // Location in the image
        bool decoded;        // True if text was extracted, False if only detected
    };

    struct BarcodeScannerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // AI Model Path (YOLO detection model trained on Barcodes/QR)
        std::string model_path;

        // Inference Input Settings
        int input_width = 640;
        int input_height = 640;

        // Detection Sensitivity
        float conf_threshold = 0.5f;
        float nms_threshold = 0.45f;

        // Attempt to decode content using OpenCV after detection?
        // If false, only returns the bounding box.
        bool enable_decoding = true;
    };

    class BarcodeScanner {
    public:
        explicit BarcodeScanner(const BarcodeScannerConfig& config);
        ~BarcodeScanner();

        // Move semantics
        BarcodeScanner(BarcodeScanner&&) noexcept;
        BarcodeScanner& operator=(BarcodeScanner&&) noexcept;
        BarcodeScanner(const BarcodeScanner&) = delete;
        BarcodeScanner& operator=(const BarcodeScanner&) = delete;

        /**
         * @brief Scans an image for barcodes.
         *
         * 1. AI Model finds candidate regions (robust to blur/rotation).
         * 2. (Optional) Classical Decoder attempts to read the cropped region.
         */
        std::vector<BarcodeResult> scan(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision