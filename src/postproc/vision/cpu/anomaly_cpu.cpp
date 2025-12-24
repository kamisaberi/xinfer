#include "anomaly_cpu.h"
#include <xinfer/core/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace xinfer::postproc {

CpuAnomalyPostproc::CpuAnomalyPostproc() {}
CpuAnomalyPostproc::~CpuAnomalyPostproc() {}

void CpuAnomalyPostproc::init(const AnomalyConfig& config) {
    m_config = config;
}

// Helper: Reduce NCHW error map to 1-channel Heatmap
// This is CPU-bound iteration, but simple enough for NEON autovectorization
void CpuAnomalyPostproc::compute_heatmap_nchw(const float* diff_data, float* heatmap_data, 
                                              int channels, int height, int width) {
    int spatial_size = height * width;
    
    // Clear heatmap
    std::fill(heatmap_data, heatmap_data + spatial_size, 0.0f);

    // Sum across channels
    for (int c = 0; c < channels; ++c) {
        const float* channel_ptr = diff_data + (c * spatial_size);
        for (int i = 0; i < spatial_size; ++i) {
            heatmap_data[i] += channel_ptr[i];
        }
    }

    // Average
    float inv_c = 1.0f / channels;
    for (int i = 0; i < spatial_size; ++i) {
        heatmap_data[i] *= inv_c;
    }
}

AnomalyResult CpuAnomalyPostproc::process(const core::Tensor& input, 
                                          const core::Tensor& reconstruction) {
    AnomalyResult result;
    
    // 1. Validation
    if (input.size() != reconstruction.size()) {
        XINFER_LOG_ERROR("Anomaly Postproc: Input and Reconstruct sizes do not match.");
        return result;
    }

    // Shapes: [Batch, C, H, W]
    // We assume Batch=1 for simplicity in this snippet
    auto shape = input.shape();
    int channels = (int)shape[1];
    int height   = (int)shape[2];
    int width    = (int)shape[3];
    size_t total_elements = input.size();

    // 2. Compute Difference (L1 or L2)
    // We use a temporary buffer for the difference map
    std::vector<float> diff_buffer(total_elements);
    const float* in_ptr = static_cast<const float*>(input.data());
    const float* recon_ptr = static_cast<const float*>(reconstruction.data());
    
    // OpenCV's absdiff is very fast (AVX/NEON)
    // We treat the tensors as 1D arrays here for speed
    cv::Mat in_mat(1, total_elements, CV_32F, const_cast<float*>(in_ptr));
    cv::Mat recon_mat(1, total_elements, CV_32F, const_cast<float*>(recon_ptr));
    cv::Mat diff_mat(1, total_elements, CV_32F, diff_buffer.data());

    cv::absdiff(in_mat, recon_mat, diff_mat);

    // 3. Reduce to Heatmap (Multi-channel -> Single channel)
    // Result Tensor: [1, 1, H, W]
    result.heatmap.resize({1, 1, (int64_t)height, (int64_t)width}, core::DataType::kFLOAT);
    float* heatmap_ptr = static_cast<float*>(result.heatmap.data());

    compute_heatmap_nchw(diff_buffer.data(), heatmap_ptr, channels, height, width);

    // 4. Post-Process Heatmap (Blur)
    // Gaussian Blur helps smooth out single-pixel noise which is common in reconstruction
    cv::Mat heatmap_cv(height, width, CV_32F, heatmap_ptr);
    
    if (m_config.use_smoothing) {
        int k = m_config.kernel_size; 
        if (k % 2 == 0) k++; // Kernel must be odd
        cv::GaussianBlur(heatmap_cv, heatmap_cv, cv::Size(k, k), 0);
    }

    // 5. Calculate Score
    // Usually Max value in the heatmap determines the anomaly score
    double min_val, max_val;
    cv::minMaxLoc(heatmap_cv, &min_val, &max_val);
    result.anomaly_score = (float)max_val;

    // 6. Verdict
    result.is_anomaly = (result.anomaly_score > m_config.threshold);

    // 7. (Optional) Generate Binary Mask Tensor for visualization
    if (result.is_anomaly) {
        result.segmentation_mask.resize({1, 1, (int64_t)height, (int64_t)width}, core::DataType::kUINT8);
        uint8_t* mask_ptr = static_cast<uint8_t*>(result.segmentation_mask.data());
        cv::Mat mask_cv(height, width, CV_8U, mask_ptr);
        
        // Threshold: Values > thresh become 255, else 0
        cv::threshold(heatmap_cv, mask_cv, m_config.threshold, 255, cv::THRESH_BINARY);
    }

    return result;
}

} // namespace xinfer::postproc