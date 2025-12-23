#pragma once

#include <xinfer/preproc/audio/audio_preprocessor.h>
#include <xinfer/preproc/audio/types.h>

#include <vector>

// Apple Framework Headers
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

namespace xinfer::preproc {

/**
 * @brief Apple Accelerate Audio Preprocessor
 * 
 * Optimized for macOS/iOS. Uses the vDSP framework to perform
 * hardware-accelerated FFT, Windowing, and Matrix Multiplication.
 * 
 * Features:
 * - Uses vDSP_DFT_zop (Discrete Fourier Transform)
 * - Uses vDSP_mmul for fast Mel-Matrix multiplication
 * - Handles DSPSplitComplex memory layout natively
 */
class AccelerateAudioPreprocessor : public IAudioPreprocessor {
public:
    AccelerateAudioPreprocessor();
    ~AccelerateAudioPreprocessor() override;

    /**
     * @brief Initialize vDSP setup.
     * Pre-calculates Window function and Mel Basis matrix.
     */
    void init(const AudioPreprocConfig& config);

    /**
     * @brief Process Audio using vDSP.
     * 
     * @param src Input PCM data (Host pointer).
     * @param dst Output Tensor (Host pointer, usually).
     */
    void process(const AudioBuffer& src, core::Tensor& dst) override;

private:
    AudioPreprocConfig m_config;

#ifdef __APPLE__
    // --- vDSP Resources ---
    
    // Opaque setup object for DFT (Discrete Fourier Transform)
    vDSP_DFT_Setup m_dft_setup = nullptr;

    // --- Pre-computed Constants ---
    
    // Window function (Hann/Hamming) applied to time-domain audio
    std::vector<float> m_window;
    
    // Mel Filterbank Matrix (Flattened [n_mels * (n_fft/2 + 1)])
    std::vector<float> m_mel_basis;

    // --- Workspace Buffers (Reused to avoid allocation loop) ---
    
    // vDSP requires "Split Complex" format (Separate Real and Imaginary arrays)
    // We allocate these once based on n_fft size.
    std::vector<float> m_split_real;
    std::vector<float> m_split_imag;
    
    // Buffer for Power Spectrum (|X|^2)
    std::vector<float> m_magnitudes;
    
    // Buffer for a single frame of Mel features before writing to Tensor
    std::vector<float> m_mel_frame;

    // --- Helpers ---
    void build_window();
    void build_mel_basis();
#endif
};

} // namespace xinfer::preproc