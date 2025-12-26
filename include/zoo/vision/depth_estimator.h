#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Depth Estimation.
     */
    struct DepthResult {
        // Raw depth values (Float32).
        // For MiDaS, this is usually inverse relative depth (disparity).
        // Resized to match original input image resolution.
        cv::Mat depth_raw;

        // Colorized visualization (Uint8, BGR).
        // Normalized 0-255 and colormapped (INFERNO/JET).
        cv::Mat depth_vis;
    };

    struct DepthEstimatorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (.xml, .engine, .rknn, etc.)
        std::string model_path;

        // Input Specs (Model dependent, usually 384x384 or 518x518 for DepthAnything)
        int input_width = 384;
        int input_height = 384;

        // Normalization (MiDaS/DepthAnything usually use ImageNet stats)
        std::vector<float> mean = {0.485f * 255, 0.456f * 255, 0.406f * 255};
        std::vector<float> std  = {0.229f * 255, 0.224f * 255, 0.225f * 255};

        // Visualization Colormap (e.g., cv::COLORMAP_INFERNO)
        int colormap = cv::COLORMAP_INFERNO;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DepthEstimator {
    public:
        explicit DepthEstimator(const DepthEstimatorConfig& config);
        ~DepthEstimator();

        // Move semantics
        DepthEstimator(DepthEstimator&&) noexcept;
        DepthEstimator& operator=(DepthEstimator&&) noexcept;
        DepthEstimator(const DepthEstimator&) = delete;
        DepthEstimator& operator=(const DepthEstimator&) = delete;

        /**
         * @brief Estimate depth from a single image.
         *
         * @param image Input image (BGR usually).
         * @return DepthResult containing raw float data and color visualization.
         */
        DepthResult estimate(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision