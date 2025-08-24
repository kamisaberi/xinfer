#include <include/preproc/audio_processor.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) { const cudaError_t e = call; if (e != cudaSuccess) { throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(e))); } }
#define CHECK_CUFFT(call) { const cufftResult_t s = call; if (s != CUFFT_SUCCESS) { throw std::runtime_error("cuFFT Error"); } }

namespace xinfer::preproc {

__global__ void frame_and_window_kernel(const float* waveform, const float* window, float* framed_waveform, int n_fft, int hop_length, int n_frames) {
    int frame_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    if (frame_idx >= n_frames) return;

    int waveform_start_idx = frame_idx * hop_length;

    if (thread_idx < n_fft) {
        framed_waveform[frame_idx * n_fft + thread_idx] = waveform[waveform_start_idx + thread_idx] * window[thread_idx];
    }
}

__global__ void power_to_mel_log_kernel(const cufftComplex* stft_output, const float* mel_filterbank, float* mel_spectrogram, int n_fft, int hop_length, int n_frames, int n_mels) {
    int frame_idx = blockIdx.x;
    int mel_idx = threadIdx.x;
    if (frame_idx >= n_frames || mel_idx >= n_mels) return;

    int fft_bins = n_fft / 2 + 1;
    float mel_energy = 0.0f;

    for (int bin_idx = 0; bin_idx < fft_bins; ++bin_idx) {
        cufftComplex val = stft_output[frame_idx * fft_bins + bin_idx];
        float power = val.x * val.x + val.y * val.y;
        mel_energy += power * mel_filterbank[mel_idx * fft_bins + bin_idx];
    }

    mel_spectrogram[frame_idx * n_mels + mel_idx] = log10f(fmaxf(1e-10f, mel_energy));
}

struct AudioProcessor::Impl {
    AudioProcessorConfig config_;
    cufftHandle fft_plan_;

    float* d_waveform_ = nullptr;
    float* d_padded_waveform_ = nullptr;
    float* d_window_ = nullptr;
    float* d_framed_windowed_ = nullptr;
    cufftComplex* d_stft_output_ = nullptr;
    float* d_mel_filterbank_ = nullptr;

    Impl(const AudioProcessorConfig& config) : config_(config) {
        CHECK_CUFFT(cufftPlan1d(&fft_plan_, config.n_fft, CUFFT_R2C, 1));

        std::vector<float> h_window(config.n_fft);
        for (int i = 0; i < config.n_fft; ++i) {
            h_window[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / (config.n_fft - 1));
        }
        CHECK_CUDA(cudaMalloc(&d_window_, config.n_fft * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_window_, h_window.data(), config.n_fft * sizeof(float), cudaMemcpyHostToDevice));

        int fft_bins = config.n_fft / 2 + 1;
        std::vector<float> h_mel_filterbank(config.n_mels * fft_bins);
        float min_mel = 1127.0f * log1pf(config.f_min / 700.0f);
        float max_mel = 1127.0f * log1pf(config.f_max / 700.0f);
        std::vector<float> mel_points(config.n_mels + 2);
        for (int i = 0; i < config.n_mels + 2; ++i) {
            mel_points[i] = min_mel + (max_mel - min_mel) * i / (config.n_mels + 1);
        }
        std::vector<float> hz_points(config.n_mels + 2);
        for (int i = 0; i < config.n_mels + 2; ++i) {
            hz_points[i] = 700.0f * (expf(mel_points[i] / 1127.0f) - 1.0f);
        }
        std::vector<int> bin_points(config.n_mels + 2);
        for (int i = 0; i < config.n_mels + 2; ++i) {
            bin_points[i] = floorf((config.n_fft + 1) * hz_points[i] / config.sample_rate);
        }

        for (int i = 0; i < config.n_mels; ++i) {
            for (int j = 0; j < fft_bins; ++j) {
                float val = 0.0f;
                if (j >= bin_points[i] && j <= bin_points[i+1]) {
                    val = (float)(j - bin_points[i]) / (bin_points[i+1] - bin_points[i]);
                } else if (j > bin_points[i+1] && j <= bin_points[i+2]) {
                    val = (float)(bin_points[i+2] - j) / (bin_points[i+2] - bin_points[i+1]);
                }
                h_mel_filterbank[i * fft_bins + j] = val;
            }
        }
        CHECK_CUDA(cudaMalloc(&d_mel_filterbank_, config.n_mels * fft_bins * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_mel_filterbank_, h_mel_filterbank.data(), config.n_mels * fft_bins * sizeof(float), cudaMemcpyHostToDevice));
    }
    ~Impl() {
        cudaFree(d_waveform_);
        cudaFree(d_padded_waveform_);
        cudaFree(d_window_);
        cudaFree(d_framed_windowed_);
        cudaFree(d_stft_output_);
        cudaFree(d_mel_filterbank_);
        cufftDestroy(fft_plan_);
    }
};

AudioProcessor::AudioProcessor(const AudioProcessorConfig& config) : pimpl_(new Impl(config)) {}
AudioProcessor::~AudioProcessor() = default;
AudioProcessor::AudioProcessor(AudioProcessor&&) noexcept = default;
AudioProcessor& AudioProcessor::operator=(AudioProcessor&&) noexcept = default;

void AudioProcessor::process(const std::vector<float>& waveform, core::Tensor& out_spectrogram) {
    if (!pimpl_) throw std::runtime_error("AudioProcessor is in a moved-from state.");

    CHECK_CUDA(cudaMalloc(&pimpl_->d_waveform_, waveform.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(pimpl_->d_waveform_, waveform.data(), waveform.size() * sizeof(float), cudaMemcpyHostToDevice));

    int n_frames = (waveform.size() - pimpl_->config_.n_fft) / pimpl_->config_.hop_length + 1;
    CHECK_CUDA(cudaMalloc(&pimpl_->d_framed_windowed_, (size_t)n_frames * pimpl_->config_.n_fft * sizeof(float)));

    frame_and_window_kernel<<<n_frames, pimpl_->config_.n_fft>>>(pimpl_->d_waveform_, pimpl_->d_window_, pimpl_->d_framed_windowed_, pimpl_->config_.n_fft, pimpl_->config_.hop_length, n_frames);
    CHECK_CUDA(cudaGetLastError());

    int fft_bins = pimpl_->config_.n_fft / 2 + 1;
    CHECK_CUDA(cudaMalloc(&pimpl_->d_stft_output_, (size_t)n_frames * fft_bins * sizeof(cufftComplex)));
    CHECK_CUFFT(cufftSetStream(pimpl_->fft_plan_, 0));
    CHECK_CUFFT(cufftExecR2C(pimpl_->fft_plan_, pimpl_->d_framed_windowed_, pimpl_->d_stft_output_));
    CHECK_CUDA(cudaGetLastError());

    power_to_mel_log_kernel<<<n_frames, pimpl_->config_.n_mels>>>(pimpl_->d_stft_output_, pimpl_->d_mel_filterbank_, (float*)out_spectrogram.data(), pimpl_->config_.n_fft, pimpl_->config_.hop_length, n_frames, pimpl_->config_.n_mels);
    CHECK_CUDA(cudaGetLastError());

    cudaFree(pimpl_->d_waveform_);
    pimpl_->d_waveform_ = nullptr;
    cudaFree(pimpl_->d_framed_windowed_);
    pimpl_->d_framed_windowed_ = nullptr;
    cudaFree(pimpl_->d_stft_output_);
    pimpl_->d_stft_output_ = nullptr;
}

} // namespace xinfer::preproc