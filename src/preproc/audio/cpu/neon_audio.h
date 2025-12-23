#pragma once
#include <xinfer/preproc/audio/audio_preprocessor.h>
#include <xinfer/preproc/audio/types.h>
#include <vector>

namespace xinfer::preproc {

class NeonAudioPreprocessor : public IAudioPreprocessor {
public:
    void init(const AudioPreprocConfig& config);
    void process(const AudioBuffer& src, core::Tensor& dst) override;

private:
    AudioPreprocConfig m_config;
    std::vector<float> m_window;
    std::vector<float> m_mel_basis;

    void build_window();
    void build_mel_basis();

    // The NEON optimized function
    void apply_mel_filterbank_neon(const float* power_spec, float* mel_output);
};

}