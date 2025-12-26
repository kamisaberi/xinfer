#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Change Detection
     */
    struct ChangeResult {
        bool change_detected;   // True if changes exceed threshold
        float change_ratio;     // Percentage of pixels changed (0.0 to 1.0)

        // Visualization
        cv::Mat diff_mask;      // Binary mask (255 = change, 0 = no change)
        std::vector<cv::Rect> bounding_boxes; // Blobs of changed areas
    };

    enum class ChangeMethod {
        AI_SIAMESE = 0,      // Robust Feature comparison (Ignore lighting shifts)
        CLASSICAL_KNN = 1,   // Fast Background Subtraction (Static Camera)
        CLASSICAL_MOG2 = 2   // Mixture of Gaussians
    };

    struct ChangeDetectorConfig {
        // Hardware Target (for AI mode)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Mode Selection
        ChangeMethod method = ChangeMethod::AI_SIAMESE;

        // Model Path (Required only for AI_SIAMESE)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Sensitivity
        float threshold = 0.5f;        // For AI: Similarity distance threshold
        int min_area = 500;            // Minimum pixel area to count as a valid object
        double learning_rate = -1;     // For Classical: How fast background updates (-1 = auto)
    };

    class ChangeDetector {
    public:
        explicit ChangeDetector(const ChangeDetectorConfig& config);
        ~ChangeDetector();

        // Move semantics
        ChangeDetector(ChangeDetector&&) noexcept;
        ChangeDetector& operator=(ChangeDetector&&) noexcept;
        ChangeDetector(const ChangeDetector&) = delete;
        ChangeDetector& operator=(const ChangeDetector&) = delete;

        /**
         * @brief Detect changes in the current frame.
         *
         * @param image Current video frame.
         * @return Result containing mask and bounding boxes.
         */
        ChangeResult detect(const cv::Mat& image);

        /**
         * @brief Explicitly set the reference frame (For AI Siamese mode).
         * If not set, the very first frame passed to detect() becomes reference.
         */
        void set_reference(const cv::Mat& ref_image);

        /**
         * @brief Reset the background model.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision