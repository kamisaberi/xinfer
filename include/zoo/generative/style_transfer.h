#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct StyleTransferConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., van_gogh_style.onnx)
        // Each model is trained for a *single* style.
        std::string model_path;

        // Input Specs (Can be variable, but fixed is faster)
        int input_width = 512;
        int input_height = 512;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class StyleTransfer {
    public:
        explicit StyleTransfer(const StyleTransferConfig& config);
        ~StyleTransfer();

        // Move semantics
        StyleTransfer(StyleTransfer&&) noexcept;
        StyleTransfer& operator=(StyleTransfer&&) noexcept;
        StyleTransfer(const StyleTransfer&) = delete;
        StyleTransfer& operator=(const StyleTransfer&) = delete;

        /**
         * @brief Apply the learned style to a content image.
         *
         * @param content_image The photo to be transformed.
         * @return The stylized image.
         */
        cv::Mat apply(const cv::Mat& content_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative