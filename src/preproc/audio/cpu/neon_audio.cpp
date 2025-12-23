#include "neon_audio.h"
#include <xinfer/core/logging.h>
#include <cmath>
#include <cstring>
#include <algorithm>

// Use KissFFT for the FFT part (it's reasonably fast)
#include <third_party/kissfft/kiss_fftr.h>

#if defined(__aarch64__) || defined(__arm__)
#include <arm_neon.h>
#endif

namespace xinfer::preproc {

// ... (Copy build_window and build_mel_basis from cpu_audio.cpp) ...

#if defined(__aarch64__) || defined(__arm__)

/**
 * @brief NEON Optimized Matrix Multiply for Mel Filterbank
 * Computes: Mel_Frame = Mel_Basis * Power_Spectrum
 */
void NeonAudioPreprocessor::apply_mel_filterbank_neon(const float* power_spec, float* mel_output) {
    int n_mels = m_config.n_mels;
    int spec_size = m_config.n_fft / 2 + 1;

    for (int m = 0; m < n_mels; ++m) {
        float32x4_t v_sum = vdupq_n_f32(0.0f); // Accumulator = [0, 0, 0, 0]
        
        const float* row_basis = &m_mel_basis[m * spec_size];
        
        int k = 0;
        // Process 4 bins at a time
        for (; k <= spec_size - 4; k += 4) {
            float32x4_t v_basis = vld1q_f32(row_basis + k);
            float32x4_t v_spec  = vld1q_f32(power_spec + k);
            
            // Fused Multiply-Add: sum += basis * spec
            v_sum = vmlaq_f32(v_sum, v_basis, v_spec);
        }

        // Horizontal Add: Sum the 4 lanes into one scalar
        float sum = vaddvq_f32(v_sum); 

        // Handle leftovers (Scalar loop)
        for (; k < spec_size; ++k) {
            sum += row_basis[k] * power_spec[k];
        }

        // Log Scale
        if (m_config.log_mel) {
            sum = std::log(std::max(1e-5f, sum));
        }

        mel_output[m] = sum;
    }
}

void NeonAudioPreprocessor::process(const AudioBuffer& src, core::Tensor& dst) {
    // 1. Setup KissFFT
    int num_frames = (src.num_samples - m_config.n_fft) / m_config.hop_length + 1;
    int spec_size = m_config.n_fft / 2 + 1;
    
    dst.resize({1, (int64_t)m_config.n_mels, (int64_t)num_frames}, core::DataType::kFLOAT);
    float* dst_ptr = static_cast<float*>(dst.data());

    kiss_fftr_cfg cfg = kiss_fftr_alloc(m_config.n_fft, 0, nullptr, nullptr);
    std::vector<float> time_buf(m_config.n_fft);
    std::vector<kiss_fft_cpx> freq_buf(spec_size);
    std::vector<float> power_spec(spec_size);

    for (int t = 0; t < num_frames; ++t) {
        int start = t * m_config.hop_length;
        
        // Windowing (Scalar is usually fine here, or optimize with vmulq_f32)
        for (int i = 0; i < m_config.n_fft; ++i) {
            if (start + i < src.num_samples) time_buf[i] = src.pcm_data[start + i] * m_window[i];
            else time_buf[i] = 0.0f;
        }

        kiss_fftr(cfg, time_buf.data(), freq_buf.data());

        // Compute Power
        for (int i = 0; i < spec_size; ++i) {
            power_spec[i] = (freq_buf[i].r * freq_buf[i].r + freq_buf[i].i * freq_buf[i].i) / m_config.n_fft;
        }

        // NEON Accelerated Mel Filter
        // dst_ptr points to the start of the current column (or row depending on layout)
        // Here assuming [Mels, Frames] layout, so we stride by num_frames
        // Optimization: It's better to compute a temporary column vector then scatter
        // but for simplicity, let's just create a temp frame buffer.
        
        std::vector<float> mel_frame(m_config.n_mels);
        apply_mel_filterbank_neon(power_spec.data(), mel_frame.data());

        // Write to output tensor
        for(int m=0; m<m_config.n_mels; ++m) {
            dst_ptr[m * num_frames + t] = mel_frame[m];
        }
    }
    kiss_fftr_free(cfg);
}

#else
// Fallback for non-ARM compilation
void NeonAudioPreprocessor::process(const AudioBuffer& src, core::Tensor& dst) {
    XINFER_LOG_ERROR("NeonAudioPreprocessor requires ARM architecture.");
}
void NeonAudioPreprocessor::init(const AudioPreprocConfig& config) {}
#endif

} // namespace xinfer::preproc