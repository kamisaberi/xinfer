#pragma once

#include <xinfer/postproc/vision/segmentation_interface.h>

namespace xinfer::postproc {

/**
 * @brief CPU Implementation of Semantic Segmentation Post-processing.
 * 
 * Optimized ArgMax reduction for NCHW layout.
 * Supports resizing using Nearest Neighbor interpolation (to preserve class IDs).
 */
class CpuSegmentationPostproc : public ISegmentationPostprocessor {
public:
    CpuSegmentationPostproc();
    ~CpuSegmentationPostproc() override;

    void init(const SegmentationConfig& config) override;

    SegmentationResult process(const core::Tensor& logits) override;

private:
    SegmentationConfig m_config;

    /**
     * @brief Optimized ArgMax loop.
     * 
     * Iterates through channel planes to find the max index for each spatial pixel.
     * 
     * @param src Pointer to float logits (NCHW)
     * @param dst Pointer to output uint8 mask (HW)
     * @param channels Number of classes
     * @param spatial_size Height * Width
     */
    void run_argmax_nchw(const float* src, uint8_t* dst, int channels, int spatial_size);
};

} // namespace xinfer::postproc