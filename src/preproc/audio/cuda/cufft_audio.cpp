#include "cufft_audio.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <xinfer/core/logging.h>

namespace xinfer::preproc {

// --- CUDA Kernels ---

// 1. Apply Window Function & Layout for cuFFT
__global__ void apply_window_kernel(const float* __restrict__ pcm,
                                    float* __restrict__ fft_input,
                                    const float* __restrict__ window,
                                    int n_fft, int hop_len, int num_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_frames * n_fft) return;

    int frame_idx = idx / n_fft;
    int sample_idx = idx % n_fft;
    
    int pcm_idx = frame_idx * hop_len + sample_idx;
    
    // R2C cuFFT expects contiguous float array
    fft_input[idx] = pcm[pcm_idx] * window[sample_idx];
}

// 2. Magnitude -> Power -> Mel -> Log
// This kernel does the heavy matrix multiplication if Mel Basis is small enough 
// to fit in Shared Memory, or uses global memory for large banks.
// For simplicity, this is a naive global memory implementation.
__global__ void mel_filter_log_kernel(const cufftComplex* __restrict__ fft_out,
                                      float* __restrict__ mel_out,
                                      const float* __restrict__ mel_basis,
                                      int n_fft, int n_mels, int num_frames) {
    int t = blockIdx.x * blockDim.x + threadIdx.x; // Time frame
    int m = blockIdx.y * blockDim.y + threadIdx.y; // Mel bin

    if (t >= num_frames || m >= n_mels) return;

    int spec_size = n_fft / 2 + 1;
    float energy = 0.0f;

    // Dot product: MelRow . PowerSpecCol
    for (int k = 0; k < spec_size; ++k) {
        cufftComplex c = fft_out[t * spec_size + k];
        float power = (c.x * c.x + c.y * c.y) / n_fft;
        
        energy += mel_basis[m * spec_size + k] * power;
    }

    // Log (Natural Log)
    mel_out[m * num_frames + t] = logf(fmaxf(1e-5f, energy));
}

// --- Class Implementation ---

void CuFFTAudioPreprocessor::process(const AudioBuffer& src, core::Tensor& dst) {
    int num_frames = (src.num_samples - m_config.n_fft) / m_config.hop_length + 1;
    int spec_size = m_config.n_fft / 2 + 1;

    // 1. Allocations (Ideally cached)
    float* d_pcm;
    float* d_window;
    float* d_fft_in;
    cufftComplex* d_fft_out;
    float* d_mel_basis;
    
    cudaMalloc(&d_pcm, src.num_samples * sizeof(float));
    cudaMemcpy(d_pcm, src.pcm_data, src.num_samples * sizeof(float), cudaMemcpyHostToDevice);
    
    // ... (Alloc other buffers, copy Window/MelBasis from Host m_cache) ...

    // 2. Windowing
    int total_samples = num_frames * m_config.n_fft;
    apply_window_kernel<<<(total_samples + 255)/256, 256>>>(
        d_pcm, d_fft_in, d_window, m_config.n_fft, m_config.hop_length, num_frames
    );

    // 3. cuFFT Batch Execution
    cufftHandle plan;
    cufftPlan1d(&plan, m_config.n_fft, CUFFT_R2C, num_frames);
    cufftExecR2C(plan, d_fft_in, d_fft_out);

    // 4. Mel Filter + Log
    dim3 grid((num_frames + 31)/32, (m_config.n_mels + 31)/32);
    dim3 block(32, 32);
    mel_filter_log_kernel<<<grid, block>>>(
        d_fft_out, dst.data(), d_mel_basis, m_config.n_fft, m_config.n_mels, num_frames
    );

    // Cleanup
    cufftDestroy(plan);
    cudaFree(d_pcm); // ... free others
}

} // namespace xinfer::preproc