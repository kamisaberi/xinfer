#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::postproc {

struct AnomalyConfig {
    float threshold = 0.5f;     // Score above this is an anomaly
    bool use_smoothing = true;  // Apply Gaussian Blur
    int kernel_size = 7;        // Blur kernel size
};

struct AnomalyResult {
    bool is_anomaly = false;
    float anomaly_score = 0.0f;
    core::Tensor heatmap;           // Float32 [0..1]
    core::Tensor segmentation_mask; // Uint8 [0 or 255]
};

class IAnomalyPostprocessor {
public:
    virtual ~IAnomalyPostprocessor() = default;
    virtual void init(const AnomalyConfig& config) = 0;
    virtual AnomalyResult process(const core::Tensor& input, 
                                  const core::Tensor& reconstruction) = 0;
};

}