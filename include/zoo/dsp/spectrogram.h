#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::dsp {

    struct SpectrogramConfig {
        int sample_rate = 22050;
        int n_fft = 1024;
        int hop_length = 256;
    };

    class Spectrogram {
    public:
        explicit Spectrogram(const SpectrogramConfig& config);
        ~Spectrogram();

        Spectrogram(const Spectrogram&) = delete;
        Spectrogram& operator=(const Spectrogram&) = delete;
        Spectrogram(Spectrogram&&) noexcept;
        Spectrogram& operator=(Spectrogram&&) noexcept;

        cv::Mat process(const std::vector<float>& waveform);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::dsp

