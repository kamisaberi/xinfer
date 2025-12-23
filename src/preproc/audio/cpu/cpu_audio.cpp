#include "cpu_audio.h"
#include <xinfer/core/logging.h>
#include <cmath>
#include <algorithm>
#include <cstring>

// Vendored KissFFT (Header only mode)
#include <third_party/kissfft/kiss_fftr.h> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xinfer::preproc {

CpuAudioPreprocessor::CpuAudioPreprocessor() {}
CpuAudioPreprocessor::~CpuAudioPreprocessor() {}

void CpuAudioPreprocessor::init(const AudioPreprocConfig& config) {
    m_config = config;
    build_window();
    if (m_config.feature_type == AudioFeatureType::MEL_SPECTROGRAM) {
        build_mel_basis();
    }
}

void CpuAudioPreprocessor::build_window() {
    m_window.resize(m_config.n_fft);
    for (int i = 0; i < m_config.n_fft; ++i) {
        // Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
        m_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (m_config.n_fft - 1)));
    }
}

// Convert Hz to Mel
static float hz_to_mel(float freq) {
    return 2595.0f * std::log10(1.0f + freq / 700.0f);
}

// Convert Mel to Hz
static float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

void CpuAudioPreprocessor::build_mel_basis() {
    int n_fft = m_config.n_fft;
    int n_mels = m_config.n_mels;
    int sample_rate = m_config.sample_rate;
    int num_spectrogram_bins = n_fft / 2 + 1;

    float mel_min = hz_to_mel(m_config.fmin);
    float mel_max = hz_to_mel(m_config.fmax);

    // Create Mel points
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    }

    // Hz to Bin index
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = std::floor((n_fft + 1) * mel_points[i] / sample_rate);
    }

    m_mel_basis.assign(n_mels * num_spectrogram_bins, 0.0f);

    for (int i = 0; i < n_mels; ++i) {
        for (int j = bin_points[i]; j < bin_points[i + 1]; ++j) {
            m_mel_basis[i * num_spectrogram_bins + j] = 
                (float)(j - bin_points[i]) / (bin_points[i + 1] - bin_points[i]);
        }
        for (int j = bin_points[i + 1]; j < bin_points[i + 2]; ++j) {
            m_mel_basis[i * num_spectrogram_bins + j] = 
                (float)(bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1]);
        }
    }
}

void CpuAudioPreprocessor::process(const AudioBuffer& src, core::Tensor& dst) {
    // 1. Calculate dimensions
    int num_frames = (src.num_samples - m_config.n_fft) / m_config.hop_length + 1;
    int spec_size = m_config.n_fft / 2 + 1; // Real FFT output size
    
    // Resize Output Tensor
    // Layout: [1, n_mels, num_frames] (Standard for Audio Models)
    dst.resize({1, (int64_t)m_config.n_mels, (int64_t)num_frames}, core::DataType::kFLOAT);
    float* dst_ptr = static_cast<float*>(dst.data());
    std::fill(dst_ptr, dst_ptr + dst.size(), 0.0f);

    // 2. Setup FFT
    kiss_fftr_cfg cfg = kiss_fftr_alloc(m_config.n_fft, 0, nullptr, nullptr);
    std::vector<float> time_buffer(m_config.n_fft);
    std::vector<kiss_fft_cpx> freq_buffer(spec_size);

    // 3. STFT Loop
    for (int t = 0; t < num_frames; ++t) {
        int start_idx = t * m_config.hop_length;
        
        // Windowing
        for (int i = 0; i < m_config.n_fft; ++i) {
            if (start_idx + i < src.num_samples) {
                time_buffer[i] = src.pcm_data[start_idx + i] * m_window[i];
            } else {
                time_buffer[i] = 0.0f; // Padding
            }
        }

        // Execute FFT
        kiss_fftr(cfg, time_buffer.data(), freq_buffer.data());

        // Compute Power Spectrum (|X|^2)
        std::vector<float> power_spec(spec_size);
        for (int i = 0; i < spec_size; ++i) {
            float re = freq_buffer[i].r;
            float im = freq_buffer[i].i;
            power_spec[i] = (re*re + im*im) / m_config.n_fft; 
        }

        // Apply Mel Filterbank (Matrix Mul: MelBasis x PowerSpec)
        for (int m = 0; m < m_config.n_mels; ++m) {
            float val = 0.0f;
            for (int k = 0; k < spec_size; ++k) {
                val += m_mel_basis[m * spec_size + k] * power_spec[k];
            }

            // Log Scale (Log Mel Spectrogram)
            // Librosa uses: 10 * log10(val + 1e-10) usually, or ln(1+x)
            if (m_config.log_mel) {
                val = std::log(std::max(1e-5f, val)); 
            }

            // Write to Output (Column-major or Row-major depending on model)
            // Here assuming [Mels, Frames]
            dst_ptr[m * num_frames + t] = val;
        }
    }

    kiss_fftr_free(cfg);
}

} // namespace xinfer::preproc