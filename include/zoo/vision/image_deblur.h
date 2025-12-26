#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Image Deblurring.
     */
    struct DeblurResult {
        cv::Mat sharp_image; // The restored image (BGR, uint8)
        float inference_time_ms;
    };

    struct ImageDeblurConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., nafnet_gopro.onnx)
        std::string model_path;

        // Input Specs
        // Restoration models often work on fixed patches (e.g. 256x256)
        // or full resolution depending on memory.
        int input_width = 256;
        int input_height = 256;

        // Normalization (Standard is usually [0, 1] range)
        // Mean=0, Std=1, Scale=1/255
        std::vector<float> mean = {0.0f, 0.0f, 0.0f};
        std::vector<float> std  = {1.0f, 1.0f, 1.0f};
        float scale_factor = 1.0f / 255.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ImageDeblur {
    public:
        explicit ImageDeblur(const ImageDeblurConfig& config);
        ~ImageDeblur();

        // Move semantics
        ImageDeblur(ImageDeblur&&) noexcept;
        ImageDeblur& operator=(ImageDeblur&&) noexcept;
        ImageDeblur(const ImageDeblur&) = delete;
        ImageDeblur& operator=(const ImageDeblur&) = delete;

        /**
         * @brief Deblur an image.
         *
         * @param blurry_image Input image (BGR).
         * @return DeblurResult containing the sharp image.
         */
        DeblurResult process(const cv::Mat& blurry_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision