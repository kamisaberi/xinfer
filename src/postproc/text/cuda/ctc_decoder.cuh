#pragma once

#include <xinfer/postproc/text/ocr_interface.h>
#include <cuda_runtime.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CUDA Implementation of CTC Greedy Decoder.
 * 
 * Target: NVIDIA GPU (Data Center / Jetson).
 * 
 * Workflow:
 * 1. GPU Kernel: Parallel ArgMax over the vocabulary dimension.
 *    Reduces [Batch, Time, Classes] -> [Batch, Time] indices.
 * 2. D2H Copy: Transfer only the integer indices (very small) to CPU.
 * 3. CPU: Perform the lightweight "Merge Repeats & Remove Blanks" logic.
 */
class CudaCtcDecoder : public IOcrPostprocessor {
public:
    CudaCtcDecoder();
    ~CudaCtcDecoder() override;

    void init(const OcrConfig& config) override;

    std::vector<std::string> process(const core::Tensor& logits) override;

private:
    OcrConfig m_config;
    cudaStream_t m_stream = nullptr;

    // --- GPU Scratch Buffers ---
    // Stores the class index for each time step after ArgMax
    int* d_indices = nullptr;
    // Stores the confidence score for that index (optional, for thresholding)
    float* d_scores = nullptr;
    
    // Size tracking to avoid reallocations
    size_t m_capacity_elements = 0;

    // --- Host Pinned Buffer ---
    int* h_indices = nullptr;
    float* h_scores = nullptr;

    void allocate_buffers(size_t total_time_steps);
    void free_buffers();
};

} // namespace xinfer::postproc