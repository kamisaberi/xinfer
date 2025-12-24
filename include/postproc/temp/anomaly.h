#pragma once

#include <include/core/tensor.h>

namespace xinfer::postproc::anomaly
{
    /**
     * @brief Calculates the reconstruction error between an original and reconstructed image.
     *
     * This function is the core of anomaly detection with autoencoder-style models.
     * It computes the pixel-wise squared difference between the input and output,
     * generates a single anomaly score, and creates a visual anomaly map.
     * This entire process is performed efficiently on the GPU.
     *
     * @param original_image The original input tensor to the model (on GPU).
     * @param reconstructed_image The output tensor from the model (on GPU).
     * @param out_anomaly_map A pre-allocated, single-channel GPU tensor to store the heat map.
     * @param out_anomaly_score A reference to a float where the final, single anomaly score will be stored (copied to CPU).
     */
    void calculate_reconstruction_error(
        const core::Tensor& original_image,
        const core::Tensor& reconstructed_image,
        core::Tensor& out_anomaly_map,
        float& out_anomaly_score
    );
} // namespace xinfer::postproc::anomaly
