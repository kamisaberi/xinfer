#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of OCR on a single image crop.
     */
    struct OcrResult {
        std::string text;
        float confidence; // Average confidence of characters
    };

    struct OcrConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., crnn_vgg.xml, lprnet.rknn)
        std::string model_path;

        // Input Specs
        // CRNN models usually have fixed height (32) and variable or fixed width (100+)
        // LPRNet usually 94x24
        int input_width = 100;
        int input_height = 32;

        // Normalization (Standard for text: Mean=127.5, Std=127.5 -> [-1, 1])
        std::vector<float> mean = {127.5f, 127.5f, 127.5f};
        std::vector<float> std  = {127.5f, 127.5f, 127.5f};
        float scale_factor = 1.0f; // Input pixels 0-255

        // Decoder Config
        std::string vocabulary; // "0123...abc..."
        int blank_index = 0;    // CTC Blank index (often 0 or last)

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class OcrRecognizer {
    public:
        explicit OcrRecognizer(const OcrConfig& config);
        ~OcrRecognizer();

        // Move semantics
        OcrRecognizer(OcrRecognizer&&) noexcept;
        OcrRecognizer& operator=(OcrRecognizer&&) noexcept;
        OcrRecognizer(const OcrRecognizer&) = delete;
        OcrRecognizer& operator=(const OcrRecognizer&) = delete;

        /**
         * @brief Recognize text in an image.
         *
         * @param image Input image (BGR). Should be a crop of a text line.
         * @return OcrResult containing the decoded string.
         */
        OcrResult recognize(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision