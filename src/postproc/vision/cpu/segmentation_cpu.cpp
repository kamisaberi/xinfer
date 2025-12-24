#include "segmentation_cpu.h"
#include <xinfer/core/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <vector>
#include <limits>
#include <cstring>

namespace xinfer::postproc {

CpuSegmentationPostproc::CpuSegmentationPostproc() {}
CpuSegmentationPostproc::~CpuSegmentationPostproc() {}

void CpuSegmentationPostproc::init(const SegmentationConfig& config) {
    m_config = config;
}

// =================================================================================
// ArgMax Logic
// =================================================================================

void CpuSegmentationPostproc::run_argmax_nchw(const float* src, uint8_t* dst, int channels, int spatial_size) {
    // 1. Initialize Max Values with the first channel (Class 0)
    // We maintain a buffer of the current maximum *score* found so far for each pixel.
    std::vector<float> max_scores(spatial_size);
    
    // Copy channel 0 to max_scores
    std::memcpy(max_scores.data(), src, spatial_size * sizeof(float));
    
    // Initialize output mask to class 0
    std::memset(dst, 0, spatial_size * sizeof(uint8_t));

    // 2. Iterate remaining channels
    for (int c = 1; c < channels; ++c) {
        const float* class_plane = src + (c * spatial_size);
        
        // Loop can be vectorized by compiler (AVX/NEON)
        for (int i = 0; i < spatial_size; ++i) {
            float val = class_plane[i];
            if (val > max_scores[i]) {
                max_scores[i] = val;
                dst[i] = static_cast<uint8_t>(c);
            }
        }
    }
}

// =================================================================================
// Main Process
// =================================================================================

SegmentationResult CpuSegmentationPostproc::process(const core::Tensor& logits) {
    SegmentationResult result;

    // 1. Check Dimensions
    auto shape = logits.shape();
    if (shape.size() != 4) {
        XINFER_LOG_ERROR("Segmentation input must be NCHW [Batch, Classes, H, W].");
        return result;
    }

    int batch = (int)shape[0]; // Assuming Batch=1 for simplicity
    int channels = (int)shape[1];
    int height = (int)shape[2];
    int width = (int)shape[3];
    int spatial_size = height * width;

    // 2. Allocate Raw Mask Buffer (Size of Model Output)
    // We use a temporary vector or cv::Mat
    std::vector<uint8_t> raw_mask_data(spatial_size);

    // 3. Perform ArgMax
    const float* logits_ptr = static_cast<const float*>(logits.data());
    run_argmax_nchw(logits_ptr, raw_mask_data.data(), channels, spatial_size);

    // 4. Resize if needed
    // Wrap in cv::Mat
    cv::Mat raw_mask(height, width, CV_8U, raw_mask_data.data());
    cv::Mat final_mask;

    bool need_resize = (m_config.target_width > 0 && m_config.target_height > 0) &&
                       (width != m_config.target_width || height != m_config.target_height);

    if (need_resize) {
        // MUST use Nearest Neighbor to preserve class integers (don't blend Class 1 and Class 3 to get Class 2)
        cv::resize(raw_mask, final_mask, cv::Size(m_config.target_width, m_config.target_height), 0, 0, cv::INTER_NEAREST);
    } else {
        final_mask = raw_mask;
    }

    // 5. Store in Result Tensor
    int res_w = final_mask.cols;
    int res_h = final_mask.rows;
    
    result.mask.resize({1, (int64_t)res_h, (int64_t)res_w}, core::DataType::kUINT8);
    
    // Copy data (Tensor owns the memory)
    // If raw_mask was used (no resize), simple copy. If resized, copy from final_mask.
    std::memcpy(result.mask.data(), final_mask.data, res_w * res_h * sizeof(uint8_t));

    return result;
}

} // namespace xinfer::postproc