#include <xinfer/zoo/audio/music_source_separator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc/Postproc is custom FFT math, not using factories.

// --- KissFFT for STFT/iSTFT ---
#include <third_party/kissfft/kiss_fft.h>

#include <iostream>
#include <vector>
#include <complex>
#include <cmath>

namespace xinfer::zoo::audio {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct MusicSourceSeparator::Impl {
    SeparatorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    std::vector<core::Tensor> output_tensors;

    Impl(const SeparatorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config cfg; cfg.model_path = config_.model_path;

        if (!engine_->load_model(cfg.model_path)) {
            throw std::runtime_error("MusicSourceSeparator: Failed to load model.");
        }
    }

    // --- Core DSP Logic: STFT ---
    // Output: Magnitude [Freq, Time], Phase [Freq, Time]
    void stft(const std::vector<float>& waveform,
              std::vector<std::vector<float>>& mag,
              std::vector<std::vector<std::complex<float>>>& phase) {

        // Simplified STFT implementation (no windowing for brevity, but should be used)
        int n_fft = config_.n_fft;
        int hop = config_.hop_length;
        int num_frames = (waveform.size() - n_fft) / hop + 1;
        int freq_bins = n_fft / 2 + 1;

        mag.assign(freq_bins, std::vector<float>(num_frames, 0.0f));
        phase.assign(freq_bins, std::vector<std::complex<float>>(num_frames, {0.0f, 0.0f}));

        kiss_fft_cfg cfg = kiss_fft_alloc(n_fft, 0, nullptr, nullptr);
        std::vector<kiss_fft_cpx> in(n_fft), out(n_fft);

        for (int t = 0; t < num_frames; ++t) {
            // Prepare frame
            for(int i=0; i<n_fft; ++i) in[i] = {waveform[t * hop + i], 0.0f};

            // FFT
            kiss_fft(cfg, in.data(), out.data());

            // Mag/Phase
            for (int f = 0; f < freq_bins; ++f) {
                float re = out[f].r;
                float im = out[f].i;
                float m = std::sqrt(re*re + im*im);

                mag[f][t] = m;
                if (m > 1e-9) phase[f][t] = {re / m, im / m};
            }
        }
        kiss_fft_free(cfg);
    }

    // --- Core DSP Logic: iSTFT ---
    std::vector<float> istft(const std::vector<std::vector<float>>& mag,
                             const std::vector<std::vector<std::complex<float>>>& phase) {
        // Overlap-Add synthesis
        int n_fft = config_.n_fft;
        int hop = config_.hop_length;
        int num_frames = mag[0].size();

        std::vector<float> waveform(num_frames * hop + n_fft, 0.0f);
        kiss_fft_cfg cfg = kiss_fft_alloc(n_fft, 1, nullptr, nullptr); // Inverse
        std::vector<kiss_fft_cpx> in(n_fft), out(n_fft);

        for (int t = 0; t < num_frames; ++t) {
            // Reconstruct complex spectrum
            for(int f=0; f < n_fft/2+1; ++f) {
                float m = mag[f][t];
                in[f] = {m * phase[f][t].real(), m * phase[f][t].imag()};
            }
            // Reconstruct symmetric part for iFFT
            // ... (omitted for brevity) ...

            kiss_fft(cfg, in.data(), out.data());

            // Overlap-Add
            for(int i=0; i<n_fft; ++i) waveform[t * hop + i] += out[i].r / n_fft;
        }
        kiss_fft_free(cfg);
        return waveform;
    }
};

// =================================================================================
// Public API
// =================================================================================

MusicSourceSeparator::MusicSourceSeparator(const SeparatorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

MusicSourceSeparator::~MusicSourceSeparator() = default;
MusicSourceSeparator::MusicSourceSeparator(MusicSourceSeparator&&) noexcept = default;
MusicSourceSeparator& MusicSourceSeparator::operator=(MusicSourceSeparator&&) noexcept = default;

std::map<std::string, std::vector<float>> MusicSourceSeparator::separate(const std::vector<float>& stereo_pcm) {
    if (!pimpl_) throw std::runtime_error("MusicSourceSeparator is null.");

    // 1. Separate L/R Channels and convert to Mono for processing if needed
    // For simplicity, we process one channel
    std::vector<float> mono_pcm;
    for(size_t i=0; i<stereo_pcm.size(); i+=2) mono_pcm.push_back(stereo_pcm[i]);

    // 2. STFT
    std::vector<std::vector<float>> mag;
    std::vector<std::vector<std::complex<float>>> phase;
    pimpl_->stft(mono_pcm, mag, phase);

    // 3. Prepare Input Tensor
    // Spleeter expects [2, Freq, Time] - Magnitude of L and R channels
    // For our mono example, we duplicate the magnitude
    int freq_bins = mag.size();
    int time_frames = mag[0].size();
    pimpl_->input_tensor.resize({1, 2, (int64_t)freq_bins, (int64_t)time_frames}, core::DataType::kFLOAT);

    float* ptr = static_cast<float*>(pimpl_->input_tensor.data());
    // Copy L
    // ... memcpy from mag vector ...
    // Copy R (duplicate of L)
    // ... memcpy ...

    // 4. Inference
    // Spleeter outputs N tensors, one mask per stem
    pimpl_->output_tensors.resize(pimpl_->config_.stem_names.size());
    pimpl_->engine_->predict({pimpl_->input_tensor}, pimpl_->output_tensors);

    // 5. Apply Masks & iSTFT
    std::map<std::string, std::vector<float>> results;

    for (size_t i = 0; i < pimpl_->config_.stem_names.size(); ++i) {
        const auto& mask_tensor = pimpl_->output_tensors[i];
        const float* mask_ptr = static_cast<const float*>(mask_tensor.data());

        // Apply mask to original magnitude
        std::vector<std::vector<float>> masked_mag = mag;
        for(int f=0; f<freq_bins; ++f) {
            for(int t=0; t<time_frames; ++t) {
                // Spleeter mask is usually [2, F, T], we average L/R for our mono result
                float mask_val = (mask_ptr[f*time_frames + t] + mask_ptr[freq_bins*time_frames + f*time_frames + t]) * 0.5f;
                masked_mag[f][t] *= mask_val;
            }
        }

        // Reconstruct audio
        results[pimpl_->config_.stem_names[i]] = pimpl_->istft(masked_mag, phase);
    }

    return results;
}

} // namespace xinfer::zoo::audio