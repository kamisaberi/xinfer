#pragma once


#include <include/core/tensor.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace xinfer::utils {

    /**
     * @struct DenormalizationParams
     * @brief Parameters for denormalizing an image tensor.
     */
    struct DenormalizationParams {
        // Standard ImageNet params: mean={0.485, 0.456, 0.406}, std={0.229, 0.224, 0.225}
        // Generative model params: mean={0.5, 0.5, 0.5}, std={0.5, 0.5, 0.5} (for [-1, 1] range)
        std::vector<float> mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> std = {0.5f, 0.5f, 0.5f};
    };

    /**
     * @brief Converts a GPU-based xInfer::core::Tensor to a CPU-based cv::Mat.
     * @param tensor The input GPU tensor. Must have 3 or 4 dimensions (C, H, W or B, C, H, W).
     * @param params The denormalization parameters to apply.
     * @return An 8-bit, 3-channel BGR cv::Mat.
     */
    cv::Mat tensor_to_mat(const core::Tensor& tensor, const DenormalizationParams& params = {});

    /**
     * @brief Saves a GPU tensor directly to an image file.
     * @param tensor The GPU tensor to save.
     * @param filepath The path to save the image file to (e.g., "output.png").
     * @param params The denormalization parameters to apply.
     * @return True on success, false on failure.
     */
    bool save_tensor_as_image(const core::Tensor& tensor, const std::string& filepath, const DenormalizationParams& params = {});

} // namespace xinfer::utils

