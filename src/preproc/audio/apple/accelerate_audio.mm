#include "accelerate_audio.h"
#include <xinfer/core/logging.h>

#import <Accelerate/Accelerate.h>
#include <cmath>
#include <algorithm>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

namespace xinfer::preproc {

// =================================================================================
// Helpers: Mel Scale Conversion
// =================================================================================

static float hz_to_mel(float freq) {
    return 2595.0f * std::log10f(1.0f + freq / 700.0f);
}

static float mel_to_hz(float mel) {
    return 700.0f * (std::powf(10.0f, mel / 2595.0f) - 1.0f);
}

// =================================================================================
// Class Implementation
// =================================================================================

AccelerateAudioPreprocessor::AccelerateAudioPreprocessor() {}

AccelerateAudioPreprocessor::~AccelerateAudioPreprocessor() {
    if (m_dft_setup) {
        vDSP_DFT_DestroySetup(m_dft_setup);
    }
}

void AccelerateAudioPreprocessor::init(const AudioPreprocConfig& config) {
    m_config = config;

    // 1. Setup DFT (Discrete Fourier Transform)
    // We use DFT instead of FFT because vDSP_DFT allows arbitrary sizes (doesn't strict power of 2)
    // though power of 2 is still faster.
    if (m_dft_setup) vDSP_DFT_DestroySetup(m_dft_setup);
    
    m_dft_setup = vDSP_DFT_zop_CreateSetup(nullptr, m_config.n_fft, vDSP_DFT_FORWARD);
    if (!m_dft_setup) {
        XINFER_LOG_ERROR("Failed to create vDSP DFT Setup.");
        return;
    }

    // 2. Pre-calculate Window
    build_window();

    // 3. Pre-calculate Mel Filterbank
    if (m_config.feature_type == AudioFeatureType::MEL_SPECTROGRAM ||
        m_config.feature_type == AudioFeatureType::MFCC) {
        build_mel_basis();
    }

    // 4. Resize Workspace Buffers
    // vDSP uses "Split Complex" (Structure of Arrays), not Interleaved (Array of Structures)
    m_split_real.resize(m_config.n_fft);
    m_split_imag.resize(m_config.n_fft);
    
    int spec_size = m_config.n_fft / 2 + 1;
    m_magnitudes.resize(spec_size);
    m_mel_frame.resize(m_config.n_mels);
}

void AccelerateAudioPreprocessor::build_window() {
    m_window.resize(m_config.n_fft);
    
    // vDSP has built-in window generators, but we do manual for consistency with other backends
    // Hann Window: 0.5 * (1 - cos(2*pi*n / (N-1)))
    // We can use vDSP_hann_window, but it uses (N) denominator sometimes.
    // Let's match the CPU generic implementation:
    for (int i = 0; i < m_config.n_fft; ++i) {
        m_window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (m_config.n_fft - 1)));
    }
}

void AccelerateAudioPreprocessor::build_mel_basis() {
    // Standard Librosa-style Mel Filterbank Construction
    int n_fft = m_config.n_fft;
    int n_mels = m_config.n_mels;
    int sample_rate = m_config.sample_rate;
    int num_spectrogram_bins = n_fft / 2 + 1;

    float mel_min = hz_to_mel(m_config.fmin);
    float mel_max = hz_to_mel(m_config.fmax);

    // Center points in Mel
    std::vector<float> mel_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        mel_points[i] = mel_to_hz(mel_min + (mel_max - mel_min) * i / (n_mels + 1));
    }

    // Hz to FFT Bin index
    std::vector<int> bin_points(n_mels + 2);
    for (int i = 0; i < n_mels + 2; ++i) {
        bin_points[i] = std::floor((n_fft + 1) * mel_points[i] / sample_rate);
    }

    // Flattened Matrix [n_mels * num_bins]
    m_mel_basis.assign(n_mels * num_spectrogram_bins, 0.0f);

    // Populate Matrix
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

void AccelerateAudioPreprocessor::process(const AudioBuffer& src, core::Tensor& dst) {
    if (!m_dft_setup) {
        XINFER_LOG_ERROR("vDSP Setup not initialized.");
        return;
    }

    int n_fft = m_config.n_fft;
    int hop = m_config.hop_length;
    int spec_size = n_fft / 2 + 1;
    
    // 1. Calculate Frame Count
    int num_frames = (src.num_samples - n_fft) / hop + 1;
    if (num_frames < 1) return;

    // 2. Resize Output Tensor
    // Shape: [1, n_mels, num_frames]
    dst.resize({1, (int64_t)m_config.n_mels, (int64_t)num_frames}, core::DataType::kFLOAT);
    float* dst_ptr = static_cast<float*>(dst.data());

    // Prepare DSPSplitComplex structure pointing to our workspace vectors
    DSPSplitComplex split_complex;
    split_complex.realp = m_split_real.data();
    split_complex.imagp = m_split_imag.data();

    // Constant for normalization (1/N)
    float fft_norm_factor = 1.0f / n_fft;
    
    // Constant for log threshold
    float min_val = 1e-5f;

    // 3. Process Loop (Frame by Frame)
    for (int t = 0; t < num_frames; ++t) {
        int start_idx = t * hop;

        // A. Copy PCM to Real buffer and Zero Imag buffer
        // Note: vDSP has fast clear, but memcpy is fine for small chunks
        memcpy(m_split_real.data(), src.pcm_data + start_idx, n_fft * sizeof(float));
        vDSP_vclr(m_split_imag.data(), 1, n_fft); // Clear imaginary part

        // B. Apply Window Function
        // vDSP_vmul(Input, 1, Window, 1, Output, 1, Length)
        vDSP_vmul(m_split_real.data(), 1, m_window.data(), 1, m_split_real.data(), 1, n_fft);

        // C. Execute DFT
        // Result is stored back in split_complex
        vDSP_DFT_Execute(m_dft_setup, 
                         m_split_real.data(), m_split_imag.data(), 
                         m_split_real.data(), m_split_imag.data());

        // D. Compute Magnitude Squared (|Re|^2 + |Im|^2)
        // vDSP_zvmags reads from split complex, writes to float array
        // NOTE: vDSP_zvmags processes 'spec_size' elements (Nyquist), ignoring symmetric part
        vDSP_zvmags(&split_complex, 1, m_magnitudes.data(), 1, spec_size);

        // E. Normalize by N (Power Spectrum = |X|^2 / N)
        vDSP_vsmul(m_magnitudes.data(), 1, &fft_norm_factor, m_magnitudes.data(), 1, spec_size);

        // F. Apply Mel Matrix (Matrix Multiply)
        // C = A * B
        // A = Mel Basis [n_mels x spec_size]
        // B = Power Spec [spec_size x 1]
        // C = Mel Frame [n_mels x 1]
        // vDSP_mmul(MatrixA, StrideA, MatrixB, StrideB, OutputC, StrideC, RowsA, ColsC, ColsA)
        vDSP_mmul(m_mel_basis.data(), 1, 
                  m_magnitudes.data(), 1, 
                  m_mel_frame.data(), 1, 
                  m_config.n_mels, 1, spec_size);

        // G. Log Transform
        if (m_config.log_mel) {
            // 1. Clip values to min_val (to avoid log(0))
            // vDSP_vmax(Input, 1, &min_val, Output, 1, Length)
            vDSP_vmax(m_mel_frame.data(), 1, &min_val, m_mel_frame.data(), 1, m_config.n_mels);

            // 2. Natural Log (ln)
            // vvlogf(Output, Input, &Length) -> Note signature is specific to Accelerate vecLib
            int n = m_config.n_mels;
            vvlogf(m_mel_frame.data(), m_mel_frame.data(), &n);
        }

        // H. Write to Output Tensor
        // Layout: [Mels, Frames] -> we write column t
        // dst is flattened, so index is: m * num_frames + t
        for (int m = 0; m < m_config.n_mels; ++m) {
            dst_ptr[m * num_frames + t] = m_mel_frame[m];
        }
    }
}

} // namespace xinfer::preproc