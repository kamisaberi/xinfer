#pragma once

#include <xinfer/preproc/audio/audio_preprocessor.h>
#include <xinfer/preproc/audio/types.h>
#include <vector>
#include <complex>

namespace xinfer::preproc {

class CpuAudioPreprocessor : public IAudioPreprocessor {
public:
    CpuAudioPreprocessor();
    ~CpuAudioPreprocessor() override;

    void init(const AudioPreprocConfig& config);
    void process(const AudioBuffer& src, core::Tensor& dst) override;

private:
    AudioPreprocConfig m_config;

    // Pre-computed Mel Filterbank Matrix [n_mels, n_fft/2 + 1]
    std::vector<float> m_mel_basis;
    
    // Hann Window
    std::vector<float> m_window;

    // Helper: Initialize Mel Matrix (Mirrors librosa.filters.mel)
    void build_mel_basis();
    
    // Helper: Initialize Window (Mirrors scipy.signal.get_window)
    void build_window();
};

} // namespace xinfer::preproc