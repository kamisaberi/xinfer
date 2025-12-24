#include "segmentation_cuda.cuh"
#include <xinfer/core/logging.h>
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernel: Fused ArgMax + Resize
// =================================================================================

/**
 * @brief Performs ArgMax on NCHW logits while upsampling to target size.
 * 
 * Grid: Covers the TARGET dimensions (dst_w, dst_h).
 * 
 * Strategy: 
 * 1. Each thread corresponds to one pixel in the FINAL mask.
 * 2. It calculates the corresponding Nearest Neighbor coordinate in the Source logits.
 * 3. It iterates through the channels (classes) at that source coordinate to find the max.
 */
__global__ void argmax_resize_nchw_kernel(const float* __restrict__ logits,
                                          uint8_t* __restrict__ output_mask,
                                          int src_c, int src_h, int src_w,
                                          int dst_h, int dst_w) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Target X
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Target Y

    if (x >= dst_w || y >= dst_h) return;

    // 1. Map Target Coordinate -> Source Coordinate (Nearest Neighbor)
    // Formula: floor(dst_coord * scale)
    // We add 0.5f for center alignment if desired, but integer scaling is standard NN
    // scale = src / dst
    
    int sx = (x * src_w) / dst_w; 
    int sy = (y * src_h) / dst_h;
    
    // Clamp checks (usually not needed if math is right, but safe)
    if (sx >= src_w) sx = src_w - 1;
    if (sy >= src_h) sy = src_h - 1;

    int src_spatial_idx = sy * src_w + sx;
    int src_plane_size = src_h * src_w;

    // 2. ArgMax Loop
    // Iterate through all class planes at this spatial location
    float max_val = -1e30f; // -Infinity
    uint8_t max_class = 0;

    for (int c = 0; c < src_c; ++c) {
        // NCHW indexing: value at [c, sy, sx]
        // Offset = c * (H*W) + (y*W + x)
        float val = logits[c * src_plane_size + src_spatial_idx];
        
        if (val > max_val) {
            max_val = val;
            max_class = (uint8_t)c;
        }
    }

    // 3. Write Output
    output_mask[y * dst_w + x] = max_class;
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaSegmentationPostproc::CudaSegmentationPostproc() {
    cudaStreamCreate(&m_stream);
}

CudaSegmentationPostproc::~CudaSegmentationPostproc() {
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaSegmentationPostproc::init(const SegmentationConfig& config) {
    m_config = config;
}

SegmentationResult CudaSegmentationPostproc::process(const core::Tensor& logits) {
    SegmentationResult result;

    // 1. Check Dimensions
    auto shape = logits.shape(); // [Batch, Classes, H, W]
    if (shape.size() != 4) {
        XINFER_LOG_ERROR("Segmentation input must be NCHW.");
        return result;
    }

    int channels = (int)shape[1];
    int src_h = (int)shape[2];
    int src_w = (int)shape[3];

    // Determine Output Size
    int dst_w = (m_config.target_width > 0) ? m_config.target_width : src_w;
    int dst_h = (m_config.target_height > 0) ? m_config.target_height : src_h;

    // 2. Prepare Output Tensor
    // Shape: [1, H, W] (Mask)
    // Allocating directly on GPU via xInfer tensor mechanism
    result.mask.resize({1, (int64_t)dst_h, (int64_t)dst_w}, core::DataType::kUINT8);
    
    // Get Pointers
    // Assuming xInfer tensors manage device pointers correctly based on build config.
    // If logits is on CPU, we might need to copy, but for CudaPostproc we expect Device memory.
    const float* d_logits = static_cast<const float*>(logits.data());
    uint8_t* d_mask = static_cast<uint8_t*>(result.mask.data());

    if (logits.memory_type() != core::MemoryType::CudaDevice) {
        // Fallback or Error? Ideally the pipeline ensures data stays on GPU.
        XINFER_LOG_WARN_ONCE("Segmentation CUDA: Input tensor is not on GPU. Performance will suffer.");
        // (Optional: Add temporary D2D copy logic here if framework permits)
    }

    // 3. Launch Kernel
    dim3 block(32, 32);
    dim3 grid((dst_w + block.x - 1) / block.x, 
              (dst_h + block.y - 1) / block.y);

    argmax_resize_nchw_kernel<<<grid, block, 0, m_stream>>>(
        d_logits,
        d_mask,
        channels, src_h, src_w,
        dst_h, dst_w
    );

    // 4. Sync (Optional, depends on pipeline)
    // If the next step consumes this tensor on the same stream, no sync needed.
    // If copying to CPU immediately after, sync happens there.
    
    return result;
}

} // namespace xinfer::postproc