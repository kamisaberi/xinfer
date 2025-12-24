#pragma once

#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::postproc {

struct SegmentationConfig {
    int target_width = 0;  // 0 means keep model output size
    int target_height = 0;
    
    // If true, applies Softmax before Argmax (usually unnecessary for Argmax alone, 
    // but useful if you need confidence thresholding)
    bool apply_softmax = false;
};

struct SegmentationResult {
    // Integer mask where each pixel value represents the Class ID.
    // Shape: [1, Height, Width]
    // DType: kUINT8 (if classes < 256) or kINT32
    core::Tensor mask;
};

class ISegmentationPostprocessor {
public:
    virtual ~ISegmentationPostprocessor() = default;

    virtual void init(const SegmentationConfig& config) = 0;

    /**
     * @brief Process segmentation logits.
     * 
     * Expects Input Tensor: [Batch, NumClasses, Height, Width] (NCHW)
     * Performs channel-wise ArgMax.
     */
    virtual SegmentationResult process(const core::Tensor& logits) = 0;
};

} // namespace xinfer::postproc