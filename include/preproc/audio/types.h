#pragma once

#include <cstdint>
#include <vector>

namespace xinfer::preproc {

    /**
     * @brief Audio Feature Type
     * What kind of tensor should the preprocessor generate?
     */
    enum class AudioFeatureType {
        RAW_WAVEFORM = 0,   // Just normalize float[-1, 1]
        SPECTROGRAM = 1,    // STFT Magnitude
        MEL_SPECTROGRAM = 2,// Mel-scale Frequency format (Most common for Audio AI)
        MFCC = 3            // Mel-frequency cepstral coefficients (Speech Rec)
    };

    /**
     * @brief Window Function for FFT
     * Controls spectral leakage.
     */
    enum class WindowType {
        HANN = 0,     // Standard for Audio
        HAMMING = 1,  // Standard for Speech
        BLACKMAN = 2,
        RECTANGULAR = 3
    };

    /**
     * @brief Configuration for Audio Preprocessing
     *
     * Defines how to turn PCM samples into a Tensor.
     * Defaults are set to common Whisper/Speech-to-Text standards.
     */
    struct AudioPreprocConfig {
        // --- Input Specs ---
        int sample_rate = 16000;  // Expected SR (Resample if input differs)
        int num_channels = 1;     // Models usually expect Mono

        // --- Feature Extraction ---
        AudioFeatureType feature_type = AudioFeatureType::MEL_SPECTROGRAM;

        // FFT Parameters
        int n_fft = 400;          // Window size (e.g., 25ms at 16kHz)
        int hop_length = 160;     // Stride (e.g., 10ms at 16kHz)
        WindowType window_type = WindowType::HANN;

        // Mel Scale Parameters
        int n_mels = 80;          // Number of Mel bands (Height of output tensor)
        float fmin = 0.0f;
        float fmax = 8000.0f;     // Nyquist frequency usually

        // MFCC Parameters (Only if type == MFCC)
        int n_mfcc = 13;          // Standard is 13 or 40

        // Normalization
        bool log_mel = true;      // Apply Log(1 + X) to Mel Spectrogram? (Critical for AI)
    };

} // namespace xinfer::preproc