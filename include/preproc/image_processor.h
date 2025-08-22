#pragma once

#include <opencv2/opencv.hpp>
#include "../core/tensor.h"

namespace xinfer::preproc {

    /**
     * @class ImageProcessor
     * @brief Performs an entire image pre-processing pipeline in a single, fused CUDA kernel.
     */
    class ImageProcessor {
    public:
        // Configures the processing pipeline (target size, normalization values, etc.)
        ImageProcessor(int width, int height,
                       const std::vector<float>& mean,
                       const std::vector<float>& std);
        ~ImageProcessor();

        /**
         * @brief Processes a CPU image and returns a GPU tensor.
         * @param cpu_image An OpenCV BGR image.
         * @param output_tensor A pre-allocated GPU tensor to write the result into.
         */
        void process(const cv::Mat& cpu_image, core::Tensor& output_tensor);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::preproc
