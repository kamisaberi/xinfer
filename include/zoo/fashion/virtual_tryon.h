#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::fashion {

    struct TryOnConfig {
        // Hardware Target (GANs require GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // 1. Pose Estimator (e.g., OpenPose or YOLO-Pose)
        std::string pose_model_path;

        // 2. Person Segmenter (e.g., UNet)
        std::string seg_model_path;

        // 3. Geometric Warper (TPS Transformer)
        std::string warp_model_path;

        // 4. Final GAN Generator
        std::string generator_model_path;

        // --- Input Specs ---
        // Usually high-resolution for fashion
        int input_width = 768;
        int input_height = 1024;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class VirtualTryOn {
    public:
        explicit VirtualTryOn(const TryOnConfig& config);
        ~VirtualTryOn();

        // Move semantics
        VirtualTryOn(VirtualTryOn&&) noexcept;
        VirtualTryOn& operator=(VirtualTryOn&&) noexcept;
        VirtualTryOn(const VirtualTryOn&) = delete;
        VirtualTryOn& operator=(const VirtualTryOn&) = delete;

        /**
         * @brief Perform virtual try-on.
         *
         * @param person_image Image of the person.
         * @param clothing_image Image of the garment on a white background.
         * @return The composited image.
         */
        cv::Mat try_on(const cv::Mat& person_image, const cv::Mat& clothing_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::fashion