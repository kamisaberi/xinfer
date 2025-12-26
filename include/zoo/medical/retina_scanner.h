#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::medical {

    /**
     * @brief DR Grading Scale (International Clinical Diabetic Retinopathy Scale)
     */
    enum class DRGrade {
        NO_DR = 0,
        MILD = 1,
        MODERATE = 2,
        SEVERE = 3,
        PROLIFERATIVE = 4
    };

    struct RetinaResult {
        DRGrade grade;
        float confidence;

        // Heatmap for explainability (Class Activation Map - CAM)
        // Helps doctors see *why* the AI predicted a grade (e.g., hemorrhages).
        cv::Mat attention_map;

        bool refer_to_specialist; // True if Moderate or worse
    };

    struct RetinaConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., efficientnet_b3_dr.onnx)
        std::string model_path;

        // Input Specs
        // Fundus models are often trained at high res (e.g., 512x512 or higher)
        int input_width = 512;
        int input_height = 512;

        // Normalization (Standard or Ben Graham's preprocessing)
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Referral Threshold
        // If probability of Moderate+ is > 0.5, flag for referral.
        float referral_threshold = 0.5f;

        // Preprocessing: Auto-crop to circular field of view?
        bool auto_crop_fundus = true;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class RetinaScanner {
    public:
        explicit RetinaScanner(const RetinaConfig& config);
        ~RetinaScanner();

        // Move semantics
        RetinaScanner(RetinaScanner&&) noexcept;
        RetinaScanner& operator=(RetinaScanner&&) noexcept;
        RetinaScanner(const RetinaScanner&) = delete;
        RetinaScanner& operator=(const RetinaScanner&) = delete;

        /**
         * @brief Analyze a retinal fundus image.
         *
         * Pipeline:
         * 1. Crop to FOV (remove black borders).
         * 2. Ben Graham's Color Normalization (Gaussian Blur subtraction).
         * 3. Inference (Classification).
         * 4. Postprocess (Softmax + Grading).
         *
         * @param fundus_image Input image (BGR).
         * @return Diagnosis result.
         */
        RetinaResult scan(const cv::Mat& fundus_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical