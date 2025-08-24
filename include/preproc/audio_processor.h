#pragma once


#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>

namespace xinfer::preproc {

    struct AudioProcessorConfig {
        int sample_rate = 22050;
        int n_fft = 1024;
        int hop_length = 256;
        int n_mels = 80;
        float f_min = 0.0f;
        float f_max = 8000.0f;
    };

    class AudioProcessor {
    public:
        explicit AudioProcessor(const AudioProcessorConfig& config);
        ~AudioProcessor();

        AudioProcessor(const AudioProcessor&) = delete;
        AudioProcessor& operator=(const AudioProcessor&) = delete;
        AudioProcessor(AudioProcessor&&) noexcept;
        AudioProcessor& operator=(AudioProcessor&&) noexcept;

        void process(const std::vector<float>& waveform, core::Tensor& out_spectrogram);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::preproc

