#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::geospatial {

    enum class DamageLevel {
        NO_DAMAGE = 0,
        MINOR_DAMAGE = 1,
        MAJOR_DAMAGE = 2,
        DESTROYED = 3
    };

    struct DamageSite {
        cv::Rect box;
        DamageLevel level;
        float confidence;
    };

    struct AssessmentResult {
        std::vector<DamageSite> damaged_sites;
        int total_structures_affected;
        float damage_area_percent;

        // Visualization overlay
        cv::Mat annotated_image;
    };

    struct DisasterConfig {
        // Hardware Target (Batch processing on GPU is common)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., siam_unet_diff.engine)
        // A model that takes two concatenated images (Before, After)
        std::string model_path;

        // Input Specs
        int input_width = 1024;
        int input_height = 1024;

        // Label Map (for damage levels, if classified)
        std::vector<std::string> labels = {"No Damage", "Minor", "Major", "Destroyed"};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DisasterAssessor {
    public:
        explicit DisasterAssessor(const DisasterConfig& config);
        ~DisasterAssessor();

        // Move semantics
        DisasterAssessor(DisasterAssessor&&) noexcept;
        DisasterAssessor& operator=(DisasterAssessor&&) noexcept;
        DisasterAssessor(const DisasterAssessor&) = delete;
        DisasterAssessor& operator=(const DisasterAssessor&) = delete;

        /**
         * @brief Assess damage by comparing two images.
         *
         * @param pre_disaster_img The "Before" image.
         * @param post_disaster_img The "After" image.
         * @return Damage report.
         */
        AssessmentResult assess(const cv::Mat& pre_disaster_img, const cv::Mat& post_disaster_img);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial