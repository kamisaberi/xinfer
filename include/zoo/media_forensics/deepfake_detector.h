#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::media_forensics {

    /**
     * @brief Analysis result for a single face in the frame.
     */
    struct FaceAnalysis {
        postproc::BoundingBox box; // Where the face is
        bool is_fake;              // True if score > threshold
        float fake_score;          // 0.0 (Real) to 1.0 (Fake)
        std::string label;         // "Real" or "Fake"
    };

    struct DeepfakeConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Face Detector ---
        // Path to a lightweight face detector (e.g., yolov8n-face.engine)
        std::string detector_model_path;
        int det_input_size = 640;

        // --- Model 2: Deepfake Classifier ---
        // Path to the forgery detection model (e.g., efficientnet_b4_deepfake.engine)
        std::string classifier_model_path;
        int cls_input_size = 224; // Often 224, 256, or 380 depending on EfficientNet variant

        // Normalization (Standard ImageNet)
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Sensitivity
        float fake_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DeepfakeDetector {
    public:
        explicit DeepfakeDetector(const DeepfakeConfig& config);
        ~DeepfakeDetector();

        // Move semantics
        DeepfakeDetector(DeepfakeDetector&&) noexcept;
        DeepfakeDetector& operator=(DeepfakeDetector&&) noexcept;
        DeepfakeDetector(const DeepfakeDetector&) = delete;
        DeepfakeDetector& operator=(const DeepfakeDetector&) = delete;

        /**
         * @brief Detect deepfakes in a video frame.
         *
         * Pipeline:
         * 1. Detect all faces.
         * 2. Crop and Align faces (simple resizing).
         * 3. Run Deepfake Classifier on each face.
         *
         * @param image Input video frame (BGR).
         * @return List of analyzed faces.
         */
        std::vector<FaceAnalysis> analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::media_forensics