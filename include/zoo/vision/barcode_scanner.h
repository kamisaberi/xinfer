#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct Barcode {
        enum class Type { UNKNOWN, QR_CODE, CODE_128, EAN_13, UPC_A, DATA_MATRIX };
        Type type;
        std::string text;
        float confidence;
        std::vector<cv::Point2f> points;
    };

    struct BarcodeScannerConfig {
        std::string detection_engine_path;
        float confidence_threshold = 0.8f;
        int input_width = 640;
        int input_height = 640;
    };

    class BarcodeScanner {
    public:
        explicit BarcodeScanner(const BarcodeScannerConfig& config);
        ~BarcodeScanner();

        BarcodeScanner(const BarcodeScanner&) = delete;
        BarcodeScanner& operator=(const BarcodeScanner&) = delete;
        BarcodeScanner(BarcodeScanner&&) noexcept;
        BarcodeScanner& operator=(BarcodeScanner&&) noexcept;

        std::vector<Barcode> scan(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

