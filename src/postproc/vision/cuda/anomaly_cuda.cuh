#pragma once

#include <xinfer/postproc/vision/anomaly_interface.h>
#include <cuda_runtime.h>

namespace xinfer::postproc {

/**
 * @brief CUDA Implementation of Anomaly Detection.
 * 
 * Optimized for NCHW layout (standard TensorRT output).
 * 
 * Pipeline:
 * 1. Fused Kernel: |Input - Recon| -> Channel Mean -> Heatmap.
 * 2. Thrust Reduction: Find Maximum value in Heatmap (Anomaly Score).
 * 3. (Optional) Thresholding to create binary mask.
 */
class CudaAnomalyPostproc : public IAnomalyPostprocessor {
public:
    CudaAnomalyPostproc();
    ~CudaAnomalyPostproc() override;

    void init(const AnomalyConfig& config) override;

    AnomalyResult process(const core::Tensor& input, 
                          const core::Tensor& reconstruction) override;

private:
    AnomalyConfig m_config;
    cudaStream_t m_stream = nullptr;

    // Internal scratch buffer for the heatmap on GPU
    float* d_heatmap = nullptr;
    size_t d_heatmap_capacity = 0;

    // Helper to manage GPU memory allocation
    void reserve_heatmap(size_t elements);
};

} // namespace xinfer::postproc