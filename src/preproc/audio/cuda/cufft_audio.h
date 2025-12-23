#pragma once

#include <xinfer/preproc/audio/audio_preprocessor.h>
#include <xinfer/preproc/audio/types.h>

// NVIDIA Headers
#include <cuda_runtime.h>
#include <cufft.h>

#include <vector>

namespace xinfer::preproc {

/**
 * @brief CUDA Audio Preprocessor (High Throughput)
 * 
 * Uses NVIDIA cuFFT and custom CUDA kernels to perform Feature Extraction.
 * 
 * Pipeline:
 * 1. H2D Copy (if input is CPU)
 * 2. Apply Window Function (Hann/Hamming) via CUDA Kernel
 * 3. Batch FFT (R2C) via cuFFT
 * 4. Power Spectrum -> Mel Filterbank -> Log Transform via Fused Kernel
 * 
 * Optimization:
 * - Pre-calculates Mel Basis and Window on setup.
 * - Caches GPU buffers to avoid cudaMalloc per inference.
 */
class CuFFTAudioPreprocessor : public IAudioPreprocessor {
public:
    CuFFTAudioPreprocessor();
    ~CuFFTAudioPreprocessor() override;

    /**
     * @brief Configure the processor.
     * Calculates the Mel Matrix and Window function on CPU and uploads them to GPU global memory.
     */
    void init(const AudioPreprocConfig& config);

    /**
     * @brief Process Audio.
     * 
     * @param src Input Audio Buffer (Supports Host or Device pointers).
     * @param dst Output Tensor (Must be allocated on GPU, preferably via xinfer::core::Tensor).
     */
    void process(const AudioBuffer& src, core::Tensor& dst) override;

private:
    AudioPreprocConfig m_config;

    // --- GPU Constants (Uploaded once during init) ---
    float* d_window = nullptr;      // [n_fft]
    float* d_mel_basis = nullptr;   // [n_mels * (n_fft/2 + 1)]

    // --- Dynamic GPU Buffers (Reused across calls) ---
    float* d_pcm_buffer = nullptr;        // Raw Input
    float* d_fft_input = nullptr;         // Windowed Frames (Batch x n_fft)
    cufftComplex* d_fft_output = nullptr; // FFT Result (Batch x (n_fft/2 + 1))

    // --- State Management ---
    cufftHandle m_plan = 0;         // cuFFT Plan Handle
    size_t m_pcm_capacity = 0;      // Current capacity of input buffer (samples)
    int m_plan_batch_size = 0;      // What batch size is the current plan optimized for?

    // --- Helper Methods ---
    
    /**
     * @brief Allocates or resizes GPU buffers if the input audio length changes.
     * Also recreates the cuFFT plan if the number of frames changes.
     */
    void reallocate_buffers(size_t num_samples, int num_frames);

    /**
     * @brief Computes Mel Basis and Window on CPU, then cudaMemcpy to d_window/d_mel_basis.
     */
    void precompute_and_upload_constants();

    /**
     * @brief Frees all GPU resources.
     */
    void cleanup();
};

} // namespace xinfer::preproc