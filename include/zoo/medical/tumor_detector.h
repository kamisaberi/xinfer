#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::medical {

    /**
     * @brief Result of Tumor Detection.
     */
    struct TumorResult {
        postproc::BoundingBox box; // Location (x, y, w, h)
        float confidence;          // Model certainty
        int class_id;              // 0=Benign, 1=Malignant (depends on model)
        std::string label;         // Text label

        // Severity estimation based on size/confidence
        bool is_critical;
    };

    struct TumorConfig {
        // Hardware Target (NVIDIA_TRT recommended for high-res medical images)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., yolov8_brain_tumor.engine)
        std::string model_path;

        // Labels (e.g. "glioma", "meningioma", "pituitary")
        std::vector<std::string> labels;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Input Channels: 1 for MRI/CT/X-Ray, 3 for Histology/Dermoscopy
        int input_channels = 1;

        // Sensitivity
        float conf_threshold = 0.25f; // Medical usually prefers High Recall (catch everything)
        float nms_threshold = 0.45f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class TumorDetector {
    public:
        explicit TumorDetector(const TumorConfig& config);
        ~TumorDetector();

        // Move semantics
        TumorDetector(TumorDetector&&) noexcept;
        TumorDetector& operator=(TumorDetector&&) noexcept;
        TumorDetector(const TumorDetector&) = delete;
        TumorDetector& operator=(const TumorDetector&) = delete;

        /**
         * @brief Detect tumors in a medical scan slice.
         *
         * @param image Input image. Can be CV_8UC1 (Grayscale) or CV_8UC3.
         * @return List of detected anomalies.
         */
        std::vector<TumorResult> detect(const cv::Mat& image);

        /**
         * @brief Utility: Draw bounding boxes with medical alerting colors.
         */
        static void visualize(cv::Mat& image, const std::vector<TumorResult>& results);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical