#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief A single detected instance with its mask.
     */
    struct InstanceResult {
        int class_id;
        float confidence;
        std::string label;

        // Bounding Box (Scaled to original image)
        cv::Rect box;

        // Binary Mask (uint8).
        // 255 = Object, 0 = Background.
        // Note: Usually returned at network resolution (e.g. 640x640) to save memory.
        // User should resize it to 'box' dimensions if pixel-perfect overlay is needed.
        cv::Mat mask;
    };

    struct InstanceSegmenterConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8m-seg.engine)
        std::string model_path;
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Thresholds
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class InstanceSegmenter {
    public:
        explicit InstanceSegmenter(const InstanceSegmenterConfig& config);
        ~InstanceSegmenter();

        // Move semantics
        InstanceSegmenter(InstanceSegmenter&&) noexcept;
        InstanceSegmenter& operator=(InstanceSegmenter&&) noexcept;
        InstanceSegmenter(const InstanceSegmenter&) = delete;
        InstanceSegmenter& operator=(const InstanceSegmenter&) = delete;

        /**
         * @brief Segment objects in an image.
         *
         * Pipeline:
         * 1. Preprocess
         * 2. Inference (Detection Head + Mask Proto Head)
         * 3. Postprocess (Matrix Multiply Coeffs * Proto -> Sigmoid -> Crop)
         *
         * @return Vector of instances.
         */
        std::vector<InstanceResult> segment(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision