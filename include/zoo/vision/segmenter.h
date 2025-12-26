#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Semantic Segmentation.
     */
    struct SegmenterResult {
        // Raw Class ID Mask (Uint8 or Int32).
        // Dimensions match the input image resolution.
        // Value at (y,x) is the class index.
        cv::Mat mask;

        // Colorized visualization (BGR).
        // Suitable for overlaying on the original image (alpha blending).
        cv::Mat color_mask;

        float inference_time_ms;
    };

    struct SegmenterConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., deeplabv3.engine, unet.rknn)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Normalization (Standard ImageNet)
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Output Visualization
        // If empty, random colors are generated.
        // If provided, index i corresponds to class i color {R, G, B}.
        std::vector<std::vector<uint8_t>> class_colors;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Segmenter {
    public:
        explicit Segmenter(const SegmenterConfig& config);
        ~Segmenter();

        // Move semantics
        Segmenter(Segmenter&&) noexcept;
        Segmenter& operator=(Segmenter&&) noexcept;
        Segmenter(const Segmenter&) = delete;
        Segmenter& operator=(const Segmenter&) = delete;

        /**
         * @brief Segment an image.
         *
         * @param image Input image (BGR).
         * @return Result containing raw class mask and color visualization.
         */
        SegmenterResult segment(const cv::Mat& image);

        /**
         * @brief Helper to blend the mask with the original image.
         */
        static cv::Mat blend(const cv::Mat& image, const cv::Mat& mask, float alpha = 0.5f);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision