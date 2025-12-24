#include "anomaly_cuda.cuh"
#include <xinfer/core/logging.h>

// CUDA / Thrust Headers
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernels
// =================================================================================

/**
 * @brief Fused Difference & Channel Reduction Kernel
 * 
 * Computes: Heatmap[i] = Mean(Abs(Input[c, i] - Recon[c, i])) across channels.
 * 
 * Grid: 1D (Spatial Size = H * W)
 * Block: 256 threads
 */
__global__ void compute_heatmap_nchw_kernel(const float* __restrict__ input,
                                            const float* __restrict__ recon,
                                            float* __restrict__ heatmap,
                                            int channels,
                                            int spatial_size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= spatial_size) return;

    float sum_diff = 0.0f;

    // Iterate through channels for this specific pixel
    // Stride is spatial_size (Planar NCHW)
    for (int c = 0; c < channels; ++c) {
        float in_val = input[c * spatial_size + idx];
        float rec_val = recon[c * spatial_size + idx];
        
        // L1 Distance (Absolute Error)
        sum_diff += fabsf(in_val - rec_val);
    }

    // Write normalized heatmap value
    heatmap[idx] = sum_diff / (float)channels;
}

/**
 * @brief Threshold Kernel (Generate Binary Mask)
 * 
 * Mask[i] = (Heatmap[i] > Threshold) ? 255 : 0
 */
__global__ void threshold_mask_kernel(const float* __restrict__ heatmap,
                                      uint8_t* __restrict__ mask,
                                      float threshold,
                                      int size) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    mask[idx] = (heatmap[idx] > threshold) ? 255 : 0;
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaAnomalyPostproc::CudaAnomalyPostproc() {
    cudaStreamCreate(&m_stream);
}

CudaAnomalyPostproc::~CudaAnomalyPostproc() {
    if (d_heatmap) cudaFree(d_heatmap);
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaAnomalyPostproc::init(const AnomalyConfig& config) {
    m_config = config;
}

void CudaAnomalyPostproc::reserve_heatmap(size_t elements) {
    if (d_heatmap_capacity < elements) {
        if (d_heatmap) cudaFree(d_heatmap);
        cudaMalloc(&d_heatmap, elements * sizeof(float));
        d_heatmap_capacity = elements;
    }
}

AnomalyResult CudaAnomalyPostproc::process(const core::Tensor& input, 
                                           const core::Tensor& reconstruction) {
    AnomalyResult result;

    // 1. Validation
    if (input.size() != reconstruction.size()) {
        XINFER_LOG_ERROR("Anomaly CUDA: Tensor sizes mismatch.");
        return result;
    }

    auto shape = input.shape(); // [Batch, C, H, W]
    int channels = (int)shape[1];
    int height   = (int)shape[2];
    int width    = (int)shape[3];
    int spatial_size = height * width;

    // 2. Prepare Heatmap Buffer
    reserve_heatmap(spatial_size);

    // Get Raw Pointers
    const float* d_in = static_cast<const float*>(input.data());
    const float* d_rec = static_cast<const float*>(reconstruction.data());

    // 3. Launch Heatmap Kernel
    int threads = 256;
    int blocks = (spatial_size + threads - 1) / threads;

    compute_heatmap_nchw_kernel<<<blocks, threads, 0, m_stream>>>(
        d_in, d_rec, d_heatmap, channels, spatial_size
    );

    // 4. Calculate Anomaly Score (Max Value) using Thrust
    // Thrust can run on the CUDA stream directly
    thrust::device_ptr<float> thrust_ptr(d_heatmap);
    auto max_iter = thrust::max_element(thrust::cuda::par.on(m_stream), 
                                        thrust_ptr, 
                                        thrust_ptr + spatial_size);
    
    // Copy the single float result back to CPU
    float max_val;
    cudaMemcpyAsync(&max_val, max_iter.get(), sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream); // Must sync to get the score decision

    result.anomaly_score = max_val;
    result.is_anomaly = (max_val > m_config.threshold);

    // 5. Store Heatmap in Result
    // We keep the heatmap on the GPU (Tensor memory type = Device)
    result.heatmap.resize({1, 1, (int64_t)height, (int64_t)width}, core::DataType::kFLOAT);
    
    // Copy computed heatmap to result tensor
    // If Result Tensor is CPU-based, this triggers D2H copy. 
    // If it's GPU-based (ideal), it's D2D.
    cudaMemcpyAsync(result.heatmap.data(), d_heatmap, spatial_size * sizeof(float), cudaMemcpyDefault, m_stream);

    // 6. (Optional) Generate Binary Segmentation Mask
    if (result.is_anomaly) {
        result.segmentation_mask.resize({1, 1, (int64_t)height, (int64_t)width}, core::DataType::kUINT8);
        
        // We can run this on GPU too for speed
        threshold_mask_kernel<<<blocks, threads, 0, m_stream>>>(
            d_heatmap,
            static_cast<uint8_t*>(result.segmentation_mask.data()),
            m_config.threshold,
            spatial_size
        );
    }

    return result;
}

} // namespace xinfer::postproc