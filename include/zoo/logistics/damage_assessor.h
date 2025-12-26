#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::logistics {

    struct DamageLocation {
        std::string damage_type; // "Scratch", "Dent", "Crack"
        float confidence;
        postproc::BoundingBox box;
    };

    struct AssessmentResult {
        // Overall condition
        std::string overall_condition; // "Undamaged", "Minor Damage", "Severe Damage"
        float overall_confidence;

        // List of specific damages found
        std::vector<DamageLocation> damages;

        // Visualization
        cv::Mat visualization;
    };

    struct AssessorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Damage Detector (YOLO-based) ---
        // This is often a single multi-class detector
        std::string detector_model_path;
        std::string labels_path; // Labels: "Scratch", "Dent", "Crack", etc.
        int input_width = 640;
        int input_height = 640;
        float conf_threshold = 0.4f;
        float nms_threshold = 0.3f; // Damage boxes can overlap

        // --- Optional Model 2: Overall Classifier (ResNet) ---
        // If provided, runs a global classification on the full image
        std::string classifier_model_path;
        std::string classifier_labels_path;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DamageAssessor {
    public:
        explicit DamageAssessor(const AssessorConfig& config);
        ~DamageAssessor();

        // Move semantics
        DamageAssessor(DamageAssessor&&) noexcept;
        DamageAssessor& operator=(DamageAssessor&&) noexcept;
        DamageAssessor(const DamageAssessor&) = delete;
        DamageAssessor& operator=(const DamageAssessor&) = delete;

        /**
         * @brief Assess an image for damage.
         *
         * @param image Input photo of the asset.
         * @return Assessment report.
         */
        AssessmentResult assess(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::logistics