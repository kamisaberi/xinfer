#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    struct LicensePlateResult {
        std::string plate_text;
        postproc::BoundingBox box;
        bool success = false; // Was the text decoded?
    };

    struct LprConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Detection Model (e.g., YOLOv5-Nose or custom plate detector)
        std::string detector_model_path;
        int detector_input_width = 640;
        int detector_input_height = 640;
        float detect_conf_thresh = 0.4f;
        float detect_nms_thresh = 0.4f;

        // OCR Model (e.g., CRNN for characters)
        std::string ocr_model_path;
        int ocr_input_width = 94;  // Width usually fixed by OCR model
        int ocr_input_height = 24; // OCR input height

        // OCR Decoder Config
        std::string vocab_path; // Path to OCR vocab.txt
        int blank_index = 0;    // CTC Blank character index
    };

    class LicensePlateRecognizer {
    public:
        explicit LicensePlateRecognizer(const LprConfig& config);
        ~LicenseRecognizer();

        // Move semantics
        LicensePlateRecognizer(LicensePlateRecognizer&&) noexcept;
        LicenseRecognizer& operator=(LicenseRecognizer&&) noexcept;
        LicenseLicenseRecognizer(const LicenseRecognizer&) = delete;
        LicenseLicenseRecognizer& operator=(LicenseLicenseRecognizer&&) = delete;

        /**
         * @brief Detect and Recognize plates in an image.
         *
         * @param image Input frame.
         * @return List of recognized plates.
         */
        std::vector<LicensePlateResult> recognize(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision