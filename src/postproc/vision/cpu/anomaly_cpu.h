#pragma once

#include <xinfer/postproc/vision/anomaly_interface.h>
#include <xinfer/postproc/vision/types.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CPU Anomaly Detection Post-processor.
 * 
 * Optimized for: Industrial Inspection, Intrusion Detection.
 * 
 * Workflow:
 * 1. Calculate Diff: |Input - Output| or (Input - Output)^2
 * 2. Channel Reduction: Average errors across channels to create a 2D Heatmap.
 * 3. Smoothing: Gaussian Blur to remove noise.
 * 4. Scoring: Calculate Max or Mean score of the heatmap.
 */
class CpuAnomalyPostproc : public IAnomalyPostprocessor {
public:
    CpuAnomalyPostproc();
    ~CpuAnomalyPostproc() override;

    void init(const AnomalyConfig& config) override;

    /**
     * @brief Process Anomaly Map.
     * 
     * @param input The original input tensor (NCHW or NHWC).
     * @param reconstruction The model output tensor (same shape).
     * @return AnomalyResult containing score, heatmap, and binary verdict.
     */
    AnomalyResult process(const core::Tensor& input, 
                          const core::Tensor& reconstruction) override;

private:
    AnomalyConfig m_config;

    /**
     * @brief Helper to convert multi-channel difference to single-channel heatmap.
     * Handles NCHW layout typical of PyTorch/TRT models.
     */
    void compute_heatmap_nchw(const float* diff_data, float* heatmap_data, 
                              int channels, int height, int width);
};

} // namespace xinfer::postproc