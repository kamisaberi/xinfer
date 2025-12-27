#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::document {

    struct SignatureResult {
        bool signature_found;
        float confidence;
        postproc::BoundingBox box;
    };

    struct SignatureConfig {
        // Hardware Target (CPU is usually sufficient for this task)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8_signature.onnx)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.6f; // High threshold to avoid false positives

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SignatureDetector {
    public:
        explicit SignatureDetector(const SignatureConfig& config);
        ~SignatureDetector();

        // Move semantics
        SignatureDetector(SignatureDetector&&) noexcept;
        SignatureDetector& operator=(SignatureDetector&&) noexcept;
        SignatureDetector(const SignatureDetector&) = delete;
        SignatureDetector& operator=(const SignatureDetector&) = delete;

        /**
         * @brief Find signatures in a document image.
         *
         * @param image Input document scan.
         * @return A list of found signatures.
         */
        std::vector<SignatureResult> detect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::document