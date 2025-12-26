#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::insurance {

    struct PropertyAttributes {
        bool has_swimming_pool = false;
        bool has_solar_panels = false;
        bool has_overhanging_trees = false;
        bool has_trampoline = false;
    };

    struct AssessmentResult {
        // Roof Analysis
        std::string roof_condition; // "Good", "Worn", "Damaged"
        float roof_condition_confidence;
        float roof_area_sq_meters;

        // Detected Features
        PropertyAttributes attributes;

        // Visualization
        cv::Mat annotated_image;
    };

    struct AssessorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Roof Segmenter (UNet) ---
        std::string roof_seg_model_path;
        int seg_input_size = 512;

        // --- Model 2: Roof Condition Classifier (ResNet) ---
        std::string roof_cls_model_path;
        std::string roof_cls_labels_path; // "Good", "Worn", "Damaged"
        int cls_input_size = 224;

        // --- Model 3: Attribute Detector (YOLO) ---
        std::string attr_det_model_path;
        std::string attr_det_labels_path; // "Pool", "Solar_Panel", etc.

        // Calibration
        float sq_meters_per_pixel = 0.05f; // From satellite/drone altitude

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PropertyAssessor {
    public:
        explicit PropertyAssessor(const AssessorConfig& config);
        ~PropertyAssessor();

        // Move semantics
        PropertyAssessor(PropertyAssessor&&) noexcept;
        PropertyAssessor& operator=(PropertyAssessor&&) noexcept;
        PropertyAssessor(const PropertyAssessor&) = delete;
        PropertyAssessor& operator=(const PropertyAssessor&) = delete;

        /**
         * @brief Assess a property from an aerial image.
         *
         * @param image Input image (e.g., satellite or drone photo).
         * @return Structured assessment.
         */
        AssessmentResult assess(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::insurance