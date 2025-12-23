#pragma once
#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::preproc {

    struct AudioBuffer {
        const float* pcm_data; // Normalized float [-1.0, 1.0]
        size_t num_samples;
        int sample_rate;
    };

    struct AudioFeatureConfig {
        bool use_mfcc = true;
        int n_fft = 2048;
        int hop_length = 512;
        int n_mels = 128;
    };

    class IAudioPreprocessor {
    public:
        virtual ~IAudioPreprocessor() = default;

        // Raw Audio -> Spectrogram/MFCC Tensor
        virtual void process(const AudioBuffer& src, core::Tensor& dst) = 0;
    };

}