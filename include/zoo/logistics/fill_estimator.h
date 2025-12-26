#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::logistics {

    /**
     * @brief Result of Fill Estimation.
     */
    struct FillResult {
        // Percentage of the container area that is filled with cargo.
        float fill_percentage; // 0.0 to 1.0

        // Pixel area of the empty space
        float void_space_pixels;

        // Visualization overlay
        // Green = Cargo, Red = Void/Empty Space
        cv::Mat visualization;
    };

    struct FillConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., unet_cargo_segmenter.onnx)
        // A model trained to segment two classes: Cargo vs Void.
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Class Mapping (Model specific)
        int cargo_class_id = 1;
        int void_class_id = 0; // Or other index

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FillEstimator {
    public:
        explicit FillEstimator(const FillConfig& config);
        ~FillEstimator();

        // Move semantics
        FillEstimator(FillEstimator&&) noexcept;
        FillEstimator& operator=(FillEstimator&&) noexcept;
        FillEstimator(const FillEstimator&) = delete;
        FillEstimator& operator=(const FillEstimator&) = delete;

        /**
         * @brief Estimate fill level from an image.
         *
         * @param image Input image (e.g., photo of an open truck trailer).
         * @return FillResult with percentage and visualization.
         */
        FillResult estimate(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::logistics