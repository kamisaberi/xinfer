#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    /**
     * @brief A single keypoint (joint) on the body.
     */
    struct Keypoint {
        float x;
        float y;
        float confidence; // Visibility score (0.0 - 1.0)
    };

    /**
     * @brief Result of Pose Estimation.
     */
    struct PoseResult {
        // The person detection box
        postproc::BoundingBox box;

        // The skeleton (e.g., 17 points for COCO)
        // Order: Nose, L-Eye, R-Eye, L-Ear, R-Ear, L-Sho, R-Sho, ...
        std::vector<Keypoint> keypoints;
    };

    struct PoseEstimatorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8n-pose.rknn)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Thresholds
        float conf_threshold = 0.5f; // Box confidence
        float nms_threshold = 0.45f; // Box overlap
        float kpt_threshold = 0.5f;  // Keypoint visibility threshold (for visualization)

        // Model Specifics
        int num_keypoints = 17; // Standard COCO format

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PoseEstimator {
    public:
        explicit PoseEstimator(const PoseEstimatorConfig& config);
        ~PoseEstimator();

        // Move semantics
        PoseEstimator(PoseEstimator&&) noexcept;
        PoseEstimator& operator=(PoseEstimator&&) noexcept;
        PoseEstimator(const PoseEstimator&) = delete;
        PoseEstimator& operator=(const PoseEstimator&) = delete;

        /**
         * @brief Estimate poses in an image.
         *
         * Pipeline:
         * 1. Preprocess
         * 2. Inference (Single shot Box + Keypoint regression)
         * 3. Decode & NMS
         *
         * @return Vector of detected persons with keypoints.
         */
        std::vector<PoseResult> estimate(const cv::Mat& image);

        /**
         * @brief Utility: Draw skeleton on image.
         */
        static void draw_skeleton(cv::Mat& image, const std::vector<PoseResult>& results);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision