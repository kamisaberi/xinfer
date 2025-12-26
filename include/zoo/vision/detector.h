#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include the Target definition
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    struct BoundingBox {
        float x1, y1, x2, y2;
        float confidence;
        int class_id;
        std::string label;
    };

    struct DetectorConfig {
        // Platform Selection (The key to multi-platform support)
        xinfer::Target target = xinfer::Target::INTEL_OV; // Default to CPU/OpenVINO

        // Model paths
        std::string model_path;       // .engine, .rknn, .xmodel, .xml, etc.
        std::string labels_path = ""; // text file with class names

        // Hyperparameters
        float confidence_threshold = 0.45f;
        float nms_iou_threshold = 0.5f;
        int input_width = 640;
        int input_height = 640;

        // Optional: Pass hardware specifics (e.g. "CORE=0" for Rockchip)
        std::vector<std::string> vendor_params;
    };

    class ObjectDetector {
    public:
        explicit ObjectDetector(const DetectorConfig& config);
        ~ObjectDetector();

        // Move semantics
        ObjectDetector(ObjectDetector&&) noexcept;
        ObjectDetector& operator=(ObjectDetector&&) noexcept;

        // Disable copy
        ObjectDetector(const ObjectDetector&) = delete;
        ObjectDetector& operator=(const ObjectDetector&) = delete;

        /**
         * @brief Runs detection on an image.
         *
         * Internally orchestrates:
         * 1. Hardware-accelerated Preprocessing (CUDA/RGA/NEON)
         * 2. Hardware-accelerated Inference (TRT/RKNN/Vitis)
         * 3. Hardware-accelerated Postprocessing (CUDA/CPU-AVX)
         */
        std::vector<BoundingBox> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision