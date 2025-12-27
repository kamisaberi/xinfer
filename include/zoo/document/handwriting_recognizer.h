#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::document {

    /**
     * @brief Result of handwriting recognition.
     */
    struct HwrResult {
        std::string text;
        float confidence; // Sequence-level confidence
    };

    struct HwrConfig {
        // Hardware Target (CPU is often fine for single lines)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., trocr_decoder.onnx)
        // If Transformer-based, this is just the decoder part
        std::string model_path;

        // Optional: Vision Encoder for Transformer models
        std::string encoder_path;

        // --- Tokenizer/Vocab ---
        // For TrOCR, this is a BPE vocab. For CRNN, a simple character map.
        std::string vocab_path;
        int blank_index = 0; // For CTC

        // --- Input Specs ---
        int input_height = 32; // Height is usually fixed
        int input_width = 128; // Width can be fixed or dynamic

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class HandwritingRecognizer {
    public:
        explicit HandwritingRecognizer(const HwrConfig& config);
        ~HandwritingRecognizer();

        // Move semantics
        HandwritingRecognizer(HandwritingRecognizer&&) noexcept;
        HandwritingRecognizer& operator=(HandwritingRecognizer&&) noexcept;
        HandwritingRecognizer(const HandwritingRecognizer&) = delete;
        HandwritingRecognizer& operator=(const HandwritingRecognizer&) = delete;

        /**
         * @brief Recognize text in a cropped image of a handwritten line.
         *
         * @param image_line Input image (Grayscale or BGR).
         * @return The transcribed text.
         */
        HwrResult recognize(const cv::Mat& image_line);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document