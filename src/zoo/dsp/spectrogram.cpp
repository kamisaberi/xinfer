#include <xinfer/zoo/dsp/spectrogram.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We reuse the Audio Preprocessor ---
#include <xinfer/preproc/factory.h>

#include <iostream>

namespace xinfer::zoo::dsp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Spectrogram::Impl {
    SpecConfig config_;

    // The audio preprocessor does all the heavy lifting of STFT/Mel
    std::unique_ptr<preproc::IAudioPreprocessor> preproc_;

    Impl(const SpecConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Configure the preprocessor based on the desired type
        preproc_ = preproc::create_audio_preprocessor(config_.target);

        preproc::AudioPreprocConfig aud_cfg;
        aud_cfg.n_fft = config_.n_fft;
        aud_cfg.hop_length = config_.hop_length;

        if (config_.type == SpectrogramType::MEL) {
            aud_cfg.feature_type = preproc::AudioFeatureType::MEL_SPECTROGRAM;
            aud_cfg.n_mels = config_.n_mels;
        } else {
            aud_cfg.feature_type = preproc::AudioFeatureType::SPECTROGRAM;
        }

        // For pure spectrograms, log scale is often handled in visualization
        aud_cfg.log_mel = (config_.type == SpectrogramType::LOG || config_.type == SpectrogramType::MEL);

        preproc_->init(aud_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

Spectrogram::Spectrogram(const SpecConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Spectrogram::~Spectrogram() = default;
Spectrogram::Spectrogram(Spectrogram&&) noexcept = default;
Spectrogram& Spectrogram::operator=(Spectrogram&&) noexcept = default;

cv::Mat Spectrogram::generate(const std::vector<float>& signal, int sample_rate) {
    if (!pimpl_ || !pimpl_->preproc_) throw std::runtime_error("Spectrogram is null.");

    // 1. Prepare Input
    preproc::AudioBuffer buf;
    buf.pcm_data = signal.data();
    buf.num_samples = signal.size();
    buf.sample_rate = sample_rate;

    // 2. Process
    core::Tensor spec_tensor;
    pimpl_->preproc_->process(buf, spec_tensor);

    // 3. Convert Tensor -> cv::Mat
    // Tensor shape is [1, FreqBins, TimeFrames]
    auto shape = spec_tensor.shape();
    int h = (int)shape[1];
    int w = (int)shape[2];

    const float* data = static_cast<const float*>(spec_tensor.data());

    // We need to clone the data because the Tensor's memory might be reused
    cv::Mat result(h, w, CV_32F);
    std::memcpy(result.data, data, h * w * sizeof(float));

    // The preprocessor returns a row-major matrix, but spectrograms are
    // usually visualized with time on X and frequency on Y, which matches.

    return result;
}

cv::Mat Spectrogram::to_colormap(const cv::Mat& spec, bool log_scale) {
    cv::Mat to_vis = spec.clone();

    if (log_scale) {
        // Convert to decibels for better visualization
        // dB = 20 * log10(magnitude)
        // We use natural log here for simplicity and scale to 0-1
        // cv::log(to_vis + 1e-6, to_vis); // ln(x)
    }

    // Normalize to [0, 255] for colormap
    cv::Mat norm_spec;
    cv::normalize(to_vis, norm_spec, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat color_map;
    cv::applyColorMap(norm_spec, color_map, cv::COLORMAP_INFERNO);

    return color_map;
}

} // namespace xinfer::zoo::dsp