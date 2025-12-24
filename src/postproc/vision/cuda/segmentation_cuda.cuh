#pragma once

#include <xinfer/postproc/vision/segmentation_interface.h>
#include <cuda_runtime.h>

namespace xinfer::postproc {

/**
 * @brief CUDA Implementation of Semantic Segmentation Post-processing.
 * 
 * Optimized for NCHW models (UNet, DeepLab, SegFormer).
 * 
 * Key Feature: **Fused Resize & ArgMax**
 * - It computes the class ID directly onto the final target resolution.
 * - Uses Nearest Neighbor interpolation (required for segmentation masks).
 * - Avoids allocating intermediate buffers for the raw model output resizing.
 */
class CudaSegmentationPostproc : public ISegmentationPostprocessor {
public:
    CudaSegmentationPostproc();
    ~CudaSegmentationPostproc() override;

    void init(const SegmentationConfig& config) override;

    SegmentationResult process(const core::Tensor& logits) override;

private:
    SegmentationConfig m_config;
    cudaStream_t m_stream = nullptr;
};

} // namespace xinfer::postproc