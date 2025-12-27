#pragma once

#include <vector>
#include <memory>
#include <string>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::dsp {

    enum class SpectrogramType {
        LINEAR = 0, // Standard STFT
        MEL = 1,    // Mel-scaled frequency
        LOG = 2     // Logarithmic amplitude (Decibels)
    };

    struct SpecConfig {
        // Hardware Target (CPU is fine, GPU for real-time visualization)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // FFT Parameters
        int n_fft = 2048;
        int hop_length = 512;

        // Mel Scale (if type == MEL)
        int n_mels = 128;

        // Output format
        SpectrogramType type = SpectrogramType::MEL;
    };

    class Spectrogram {
    public:
        explicit Spectrogram(const SpecConfig& config);
        ~Spectrogram();

        // Move semantics
        Spectrogram(Spectrogram&&) noexcept;
        Spectrogram& operator=(Spectrogram&&) noexcept;
        Spectrogram(const Spectrogram&) = delete;
        Spectrogram& operator=(const Spectrogram&) = delete;

        /**
         * @brief Generate a spectrogram from a raw signal.
         *
         * @param signal Raw float audio/signal data.
         * @param sample_rate The sample rate of the signal.
         * @return The spectrogram as a 2D float matrix.
         */
        cv::Mat generate(const std::vector<float>& signal, int sample_rate);

        /**
         * @brief Utility to convert a float spectrogram to a color image.
         */
        static cv::Mat to_colormap(const cv::Mat& spec, bool log_scale = true);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::dsp