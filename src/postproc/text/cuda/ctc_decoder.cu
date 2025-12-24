#include "ctc_decoder.cuh"
#include <xinfer/core/logging.h>
#include <algorithm>

namespace xinfer::postproc {

// =================================================================================
// CUDA Kernel: Parallel ArgMax
// =================================================================================

/**
 * @brief Finds the max class for every (Batch, Time) pair.
 * 
 * Handles flexible strides to support both [T, B, C] and [B, T, C] layouts
 * without needing a transpose.
 */
__global__ void ctc_argmax_kernel(const float* __restrict__ logits,
                                  int* __restrict__ out_indices,
                                  float* __restrict__ out_scores,
                                  int batch_size,
                                  int time_steps,
                                  int num_classes,
                                  long long stride_batch,
                                  long long stride_time) 
{
    // Global index for the (Batch, Time) pair we are processing
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_steps = batch_size * time_steps;

    if (idx >= total_steps) return;

    // Decode linear index -> (b, t)
    // We treat the output buffer as flattened [Batch * Time] for simplicity
    int b = idx / time_steps;
    int t = idx % time_steps;

    // Calculate pointer to the start of the Class vector for this (b, t)
    // Address = Base + (b * stride_batch) + (t * stride_time)
    const float* probs = logits + (b * stride_batch) + (t * stride_time);

    // Find ArgMax
    // Note: If num_classes is huge (e.g. > 1024), this loop per thread is suboptimal.
    // For standard OCR (classes ~100) or ASR (classes ~50), this is very fast.
    float max_val = -1e30f;
    int max_idx = 0;

    for (int c = 0; c < num_classes; ++c) {
        float val = probs[c];
        if (val > max_val) {
            max_val = val;
            max_idx = c;
        }
    }

    // Write result
    out_indices[idx] = max_idx;
    out_scores[idx]  = max_val;
}

// =================================================================================
// Class Implementation
// =================================================================================

CudaCtcDecoder::CudaCtcDecoder() {
    cudaStreamCreate(&m_stream);
}

CudaCtcDecoder::~CudaCtcDecoder() {
    free_buffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

void CudaCtcDecoder::allocate_buffers(size_t total_time_steps) {
    if (total_time_steps > m_capacity_elements) {
        free_buffers();
        
        size_t int_size = total_time_steps * sizeof(int);
        size_t float_size = total_time_steps * sizeof(float);

        cudaMalloc(&d_indices, int_size);
        cudaMalloc(&d_scores, float_size);
        
        cudaMallocHost(&h_indices, int_size);
        cudaMallocHost(&h_scores, float_size);

        m_capacity_elements = total_time_steps;
    }
}

void CudaCtcDecoder::free_buffers() {
    if (d_indices) cudaFree(d_indices);
    if (d_scores) cudaFree(d_scores);
    if (h_indices) cudaFreeHost(h_indices);
    if (h_scores) cudaFreeHost(h_scores);
    d_indices = nullptr;
}

void CudaCtcDecoder::init(const OcrConfig& config) {
    m_config = config;
}

std::vector<std::string> CudaCtcDecoder::process(const core::Tensor& logits) {
    std::vector<std::string> results;

    // 1. Check Dimensions and Strides
    auto shape = logits.shape();
    if (shape.size() != 3) {
        XINFER_LOG_ERROR("CTC CUDA: Input must be 3D tensor.");
        return results;
    }

    // Heuristic for Layout (Same as CPU version)
    // [B, T, C] vs [T, B, C]
    bool batch_major = (shape[0] < shape[1]);

    int batch, time_steps, num_classes;
    long long stride_b, stride_t;

    if (batch_major) {
        // [Batch, Time, Classes]
        batch = (int)shape[0];
        time_steps = (int)shape[1];
        num_classes = (int)shape[2];
        stride_b = time_steps * num_classes;
        stride_t = num_classes;
    } else {
        // [Time, Batch, Classes]
        time_steps = (int)shape[0];
        batch = (int)shape[1];
        num_classes = (int)shape[2];
        stride_t = batch * num_classes;
        stride_b = num_classes;
    }

    // 2. Prepare Memory
    size_t total_steps = batch * time_steps;
    allocate_buffers(total_steps);

    const float* d_data = static_cast<const float*>(logits.data());

    // 3. Launch Kernel
    // This reduces the massive Float matrix to a small Int array on GPU
    int threads = 256;
    int blocks = (total_steps + threads - 1) / threads;

    ctc_argmax_kernel<<<blocks, threads, 0, m_stream>>>(
        d_data,
        d_indices,
        d_scores,
        batch,
        time_steps,
        num_classes,
        stride_b,
        stride_t
    );

    // 4. Download Results (Small transfer)
    cudaMemcpyAsync(h_indices, d_indices, total_steps * sizeof(int), cudaMemcpyDeviceToHost, m_stream);
    cudaMemcpyAsync(h_scores, d_scores, total_steps * sizeof(float), cudaMemcpyDeviceToHost, m_stream);
    cudaStreamSynchronize(m_stream);

    // 5. String Construction (CPU)
    // This part is serial but extremely fast because it iterates over Ints, not Floats
    results.reserve(batch);

    for (int b = 0; b < batch; ++b) {
        std::string decoded_str;
        int last_idx = -1;

        // Pointer to the start of this batch's timeline in the flat buffer
        // Note: Our kernel output `d_indices` is flattened logically as [Batch, Time] 
        // regardless of input layout, because we calculated `idx` based on (b * time_steps + t).
        // This effectively transposes TBC -> BTC in the output buffer automatically!
        
        const int* seq_indices = h_indices + (b * time_steps);
        const float* seq_scores = h_scores + (b * time_steps);

        for (int t = 0; t < time_steps; ++t) {
            int idx = seq_indices[t];
            float score = seq_scores[t];

            if (score >= m_config.min_confidence && idx != m_config.blank_index) {
                if (idx != last_idx) {
                    if (idx >= 0 && idx < m_config.vocabulary.size()) {
                        decoded_str += m_config.vocabulary[idx];
                    }
                }
            }
            last_idx = idx;
        }
        results.push_back(decoded_str);
    }

    return results;
}

} // namespace xinfer::postproc