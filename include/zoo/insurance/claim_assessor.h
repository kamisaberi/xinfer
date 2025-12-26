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

    enum class SeverityLevel {
        LOW = 0,
        MEDIUM = 1,
        HIGH = 2,
        TOTAL_LOSS = 3
    };

    struct DamageLocation {
        std::string type; // "Scratch", "Dent", "Broken_Glass"
        postproc::BoundingBox box;
        float confidence;
    };

    struct ClaimAssessment {
        std::string primary_peril; // e.g., "Collision", "Hail"
        float peril_confidence;
        SeverityLevel severity;
        std::vector<DamageLocation> detected_damages;
        cv::Mat visualization;
    };

    struct AssessorConfig {
        // Hardware Target (Often runs on cloud or edge servers)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Damage Detector (YOLO-based) ---
        std::string detector_model_path;
        std::string damage_labels_path; // "Scratch", "Dent", etc.
        int det_input_width = 640;
        int det_input_height = 640;
        float det_conf_thresh = 0.3f;

        // --- Model 2: Peril Classifier (ResNet) ---
        // Classifies the entire scene (Flood, Fire, etc.)
        std::string peril_model_path;
        std::string peril_labels_path;
        int peril_input_width = 224;
        int peril_input_height = 224;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ClaimAssessor {
    public:
        explicit ClaimAssessor(const AssessorConfig& config);
        ~ClaimAssessor();

        // Move semantics
        ClaimAssessor(ClaimAssessor&&) noexcept;
        ClaimAssessor& operator=(ClaimAssessor&&) noexcept;
        ClaimAssessor(const ClaimAssessor&) = delete;
        ClaimAssessor& operator=(const ClaimAssessor&) = delete;

        /**
         * @brief Assess an image from an insurance claim.
         *
         * @param image Input photo.
         * @return Structured assessment report.
         */
        ClaimAssessment assess(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::insurance