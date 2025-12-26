#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Low Light Enhancement.
     */
    struct EnhancementResult {
        cv::Mat enhanced_image; // BGR, uint8
        float inference_time_ms;
    };

    struct LowLightConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., enlightengan.onnx)
        std::string model_path;

        // Input Specs
        // Enhancement models often work on fixed patches or lower resolutions
        // to maintain speed (e.g. 512x512)
        int input_width = 512;
        int input_height = 512;

        // Normalization (Standard [0, 1] scaling usually)
        std::vector<float> mean = {0.0f, 0.0f, 0.0f};
        std::vector<float> std  = {1.0f, 1.0f, 1.0f};
        float scale_factor = 1.0f / 255.0f;

        // Post-processing Gamma Correction (optional)
        // Apply gamma > 1.0 to further brighten, < 1.0 to darken
        float post_gamma = 1.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class LowLightEnhancer {
    public:
        explicit LowLightEnhancer(const LowLightConfig& config);
        ~LowLightEnhancer();

        // Move semantics
        LowLightEnhancer(LowLightEnhancer&&) noexcept;
        LowLightEnhancer& operator=(LowLightEnhancer&&) noexcept;
        LowLightEnhancer(const LowLightEnhancer&) = delete;
        LowLightEnhancer& operator=(const LowLightEnhancer&) = delete;

        /**
         * @brief Enhance a low-light image.
         *
         * @param dark_image Input image (BGR).
         * @return EnhancementResult containing the brightened image.
         */
        EnhancementResult enhance(const cv::Mat& dark_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision