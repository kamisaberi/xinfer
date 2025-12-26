#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of an Anomaly Inspection.
     */
    struct AnomalyResult {
        bool is_anomaly;        // True if score > threshold
        float score;            // 0.0 to 1.0+ (MSE/L1 Error)

        // Visualization
        cv::Mat heatmap;        // Colorized heatmap of defects
        cv::Mat segmentation;   // Binary mask of defects
    };

    struct AnomalyConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (Autoencoder / Reconstruction Model)
        std::string model_path;

        // Input Specs
        int input_width = 256;
        int input_height = 256;

        // Normalization (Critical for Autoencoders)
        // Default: Scale 0-255 to 0-1
        std::vector<float> mean = {0.0f, 0.0f, 0.0f};
        std::vector<float> std  = {1.0f, 1.0f, 1.0f};
        float scale = 1.0f / 255.0f;

        // Detection Sensitivity
        float threshold = 0.45f;
        bool use_smoothing = true; // Apply blur to heatmap to remove noise
    };

    class AnomalyDetector {
    public:
        explicit AnomalyDetector(const AnomalyConfig& config);
        ~AnomalyDetector();

        // Move semantics
        AnomalyDetector(AnomalyDetector&&) noexcept;
        AnomalyDetector& operator=(AnomalyDetector&&) noexcept;
        AnomalyDetector(const AnomalyDetector&) = delete;
        AnomalyDetector& operator=(const AnomalyDetector&) = delete;

        /**
         * @brief Inspect an image for anomalies.
         *
         * Pipeline:
         * 1. Preprocess (Resize/Norm)
         * 2. Inference (Reconstruct image)
         * 3. Postprocess (Calculate Diff(Input, Output) -> Heatmap)
         *
         * @return Result containing score and visualization masks.
         */
        AnomalyResult inspect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision