#include "cuda_image.h"
#include <xinfer/core/logging.h>
#include <algorithm>

namespace xinfer::preproc {

// =================================================================================
// CUDA Kernels
// =================================================================================

// Structure to pass constants to kernel to avoid excessive argument lists
struct KernelParams {
    int src_w, src_h;
    int dst_w, dst_h;
    float scale_x, scale_y;
    float mean[3];
    float std[3];
    float global_scale;
    bool bgr_to_rgb;
};

/**
 * @brief Fused Kernel: Resize (Bilinear) + Normalize + HWC->NCHW
 *
 * Assumes Input is uint8 (HWC), Output is float (NCHW).
 * Optimized for coalesced writes to global memory (NCHW requires strided writes,
 * so we prioritize output layout mapping).
 */
__global__ void fused_preprocess_kernel_nchw(const uint8_t* __restrict__ src,
                                             float* __restrict__ dst,
                                             KernelParams params) {
    // Map thread (x, y) to Output Coordinate
    int dx = blockIdx.x * blockDim.x + threadIdx.x;
    int dy = blockIdx.y * blockDim.y + threadIdx.y;

    if (dx >= params.dst_w || dy >= params.dst_h) return;

    // 1. Calculate Source Coordinate (Bilinear Interpolation)
    float sx = (dx + 0.5f) * params.scale_x - 0.5f;
    float sy = (dy + 0.5f) * params.scale_y - 0.5f;

    int x_low = floorf(sx);
    int y_low = floorf(sy);
    int x_high = min(x_low + 1, params.src_w - 1);
    int y_high = min(y_low + 1, params.src_h - 1);
    x_low = max(x_low, 0);
    y_low = max(y_low, 0);

    float fx = sx - x_low;
    float fy = sy - y_low;
    float w_tl = (1.0f - fx) * (1.0f - fy);
    float w_tr = fx * (1.0f - fy);
    float w_bl = (1.0f - fx) * fy;
    float w_br = fx * fy;

    // 2. Read Source Pixels (HWC)
    // Stride is src_w * 3
    int src_stride = params.src_w * 3;

    // Pointers to the 4 neighbors
    const uint8_t* p_tl = src + y_low  * src_stride + x_low  * 3;
    const uint8_t* p_tr = src + y_low  * src_stride + x_high * 3;
    const uint8_t* p_bl = src + y_high * src_stride + x_low  * 3;
    const uint8_t* p_br = src + y_high * src_stride + x_high * 3;

    // 3. Process each channel
    // We compute all 3 channels for this pixel to save index math
    float pixel[3];

    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float val = w_tl * p_tl[c] + w_tr * p_tr[c] +
                    w_bl * p_bl[c] + w_br * p_br[c];

        // 4. Normalize
        // dst = (src * scale - mean) / std
        pixel[c] = (val * params.global_scale - params.mean[c]) / params.std[c];
    }

    // 5. Write Output (NCHW Layout)
    // Destination stride is dst_w * dst_h (Plane size)
    int plane_size = params.dst_w * params.dst_h;
    int dst_idx = dy * params.dst_w + dx;

    // Handle BGR -> RGB Swapping if needed
    if (params.bgr_to_rgb) {
        dst[dst_idx + 0 * plane_size] = pixel[2]; // R from B
        dst[dst_idx + 1 * plane_size] = pixel[1]; // G
        dst[dst_idx + 2 * plane_size] = pixel[0]; // B from R
    } else {
        dst[dst_idx + 0 * plane_size] = pixel[0];
        dst[dst_idx + 1 * plane_size] = pixel[1];
        dst[dst_idx + 2 * plane_size] = pixel[2];
    }
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaImagePreprocessor::CudaImagePreprocessor() {}

CudaImagePreprocessor::~CudaImagePreprocessor() {
    if (d_input_buffer) {
        cudaFree(d_input_buffer);
    }
}

void CudaImagePreprocessor::reserve_input_buffer(size_t size_bytes) {
    if (d_input_capacity < size_bytes) {
        if (d_input_buffer) cudaFree(d_input_buffer);
        cudaError_t err = cudaMalloc(&d_input_buffer, size_bytes);
        if (err != cudaSuccess) {
            XINFER_LOG_ERROR("CUDA Malloc failed in preprocessor");
            return;
        }
        d_input_capacity = size_bytes;
    }
}

void CudaImagePreprocessor::init(const ImagePreprocConfig& config) {
    m_config = config;
}

void CudaImagePreprocessor::process(const ImageFrame& src, core::Tensor& dst) {
    // 1. Prepare Kernel Params
    KernelParams params;
    params.src_w = src.width;
    params.src_h = src.height;
    params.dst_w = m_config.target_width;
    params.dst_h = m_config.target_height;

    params.scale_x = (float)src.width / m_config.target_width;
    params.scale_y = (float)src.height / m_config.target_height;

    // Normalization Config
    for(int i=0; i<3; ++i) {
        params.mean[i] = m_config.norm_params.mean[i];
        params.std[i]  = m_config.norm_params.std[i];
    }
    params.global_scale = m_config.norm_params.scale_factor;

    // Color Swap Check
    params.bgr_to_rgb = (src.format == ImageFormat::BGR && m_config.target_format == ImageFormat::RGB);

    // 2. Handle Memory Transfer (Host -> Device)
    const uint8_t* d_src_ptr = nullptr;

    if (src.is_device_ptr) {
        d_src_ptr = static_cast<const uint8_t*>(src.data);
    } else {
        // Slow path: Copy CPU image to GPU scratch buffer
        // Assuming 3 channels (RGB/BGR)
        size_t src_size = src.width * src.height * 3;
        reserve_input_buffer(src_size);

        cudaError_t err = cudaMemcpy(d_input_buffer, src.data, src_size, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            XINFER_LOG_ERROR("cudaMemcpy failed in preprocessor");
            return;
        }
        d_src_ptr = static_cast<const uint8_t*>(d_input_buffer);
    }

    // 3. Launch Kernel
    dim3 block(32, 32);
    dim3 grid((m_config.target_width + block.x - 1) / block.x,
              (m_config.target_height + block.y - 1) / block.y);

    float* d_dst_ptr = static_cast<float*>(dst.data());

    fused_preprocess_kernel_nchw<<<grid, block>>>(d_src_ptr, d_dst_ptr, params);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        XINFER_LOG_ERROR("CUDA Kernel launch failed: " + std::string(cudaGetErrorString(err)));
    }

    // Note: We don't synchronize here.
    // xInfer pipelines should be asynchronous. The TensorRT execution will wait on the stream implicitly.
}

} // namespace xinfer::preproc