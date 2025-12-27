#include <include/zoo/dsp/spectrogram.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) { const cudaError_t e = call; if (e != cudaSuccess) { throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(e))); } }
#define CHECK_CUFFT(call) { const cufftResult_t s = call; if (s != CUFFT_SUCCESS) { throw std::runtime_error("cuFFT Error"); } }

namespace xinfer::zoo::dsp {

__global__ void frame_and_window_kernel(const float* waveform, const float* window, float* framed_waveform, int n_fft, int hop_length, int n_frames) {
    int frame_idx = blockIdx.x;
    int thread_idx = threadIdx.x;
    if (frame_idx >= n_frames) return;

    int waveform_start_idx = frame_idx * hop_length;

    if (thread_idx < n_fft) {
        if (waveform_start_idx + thread_idx < 0) { // Should not happen with padding
             framed_waveform[frame_idx * n_fft + thread_idx] = 0.0f;
        } else {
             framed_waveform[frame_idx * n_fft + thread_idx] = waveform[waveform_start_idx + thread_idx] * window[thread_idx];
        }
    }
}

__global__ void power_to_db_kernel(const cufftComplex* stft_output, float* power_spectrogram, int n_frames, int fft_bins) {
    int frame_idx = blockIdx.x;
    int bin_idx = threadIdx.x;
    if (frame_idx >= n_frames || bin_idx >= fft_bins) return;

    cufftComplex val = stft_output[frame_idx * fft_bins + bin_idx];
    float power = val.x * val.x + val.y * val.y;
    power_spectrogram[frame_idx * fft_bins + bin_idx] = 10.0f * log10f(fmaxf(1e-10f, power));
}

struct Spectrogram::Impl {
    SpectrogramConfig config_;
    cufftHandle fft_plan_;
    float* d_window_ = nullptr;

    Impl(const SpectrogramConfig& config) : config_(config) {
        CHECK_CUFFT(cufftPlan1d(&fft_plan_, config.n_fft, CUFFT_R2C, 1));

        std::vector<float> h_window(config.n_fft);
        for (int i = 0; i < config.n_fft; ++i) {
            h_window[i] = 0.5f - 0.5f * cosf(2.0f * M_PI * i / (config.n_fft - 1));
        }
        CHECK_CUDA(cudaMalloc(&d_window_, config.n_fft * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_window_, h_window.data(), config.n_fft * sizeof(float), cudaMemcpyHostToDevice));
    }

    ~Impl() {
        cudaFree(d_window_);
        cufftDestroy(fft_plan_);
    }
};

Spectrogram::Spectrogram(const SpectrogramConfig& config) : pimpl_(new Impl(config)) {}
Spectrogram::~Spectrogram() = default;
Spectrogram::Spectrogram(Spectrogram&&) noexcept = default;
Spectrogram& Spectrogram::operator=(Spectrogram&&) noexcept = default;

cv::Mat Spectrogram::process(const std::vector<float>& waveform) {
    if (!pimpl_) throw std::runtime_error("Spectrogram is in a moved-from state.");

    int n_fft = pimpl_->config_.n_fft;
    int hop_length = pimpl_->config_.hop_length;
    int pad_amount = n_fft / 2;

    std::vector<float> padded_waveform(waveform.size() + 2 * pad_amount, 0.0f);
    std::copy(waveform.begin(), waveform.end(), padded_waveform.begin() + pad_amount);

    float* d_waveform;
    CHECK_CUDA(cudaMalloc(&d_waveform, padded_waveform.size() * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_waveform, padded_waveform.data(), padded_waveform.size() * sizeof(float), cudaMemcpyHostToDevice));

    int n_frames = (padded_waveform.size() - n_fft) / hop_length + 1;

    float* d_framed_windowed;
    CHECK_CUDA(cudaMalloc(&d_framed_windowed, (size_t)n_frames * n_fft * sizeof(float)));

    frame_and_window_kernel<<<n_frames, n_fft>>>(d_waveform + pad_amount, pimpl_->d_window_, d_framed_windowed, n_fft, hop_length, n_frames);
    CHECK_CUDA(cudaGetLastError());

    int fft_bins = n_fft / 2 + 1;
    cufftComplex* d_stft_output;
    CHECK_CUDA(cudaMalloc(&d_stft_output, (size_t)n_frames * fft_bins * sizeof(cufftComplex)));

    CHECK_CUFFT(cufftSetStream(pimpl_->fft_plan_, 0));
    CHECK_CUFFT(cufftExecR2C(pimpl_->fft_plan_, d_framed_windowed, d_stft_output));
    CHECK_CUDA(cudaGetLastError());

    float* d_power_spec;
    CHECK_CUDA(cudaMalloc(&d_power_spec, (size_t)n_frames * fft_bins * sizeof(float)));

    power_to_db_kernel<<<n_frames, fft_bins>>>(d_stft_output, d_power_spec, n_frames, fft_bins);
    CHECK_CUDA(cudaGetLastError());

    std::vector<float> h_power_spec( (size_t)n_frames * fft_bins );
    CHECK_CUDA(cudaMemcpy(h_power_spec.data(), d_power_spec, h_power_spec.size() * sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat spec_mat(fft_bins, n_frames, CV_32F, h_power_spec.data());
    cv::Mat flipped_spec;
    cv::flip(spec_mat, flipped_spec, 0); // Flip for standard visualization

    cudaFree(d_waveform);
    cudaFree(d_framed_windowed);
    cudaFree(d_stft_output);
    cudaFree(d_power_spec);

    return flipped_spec.clone();
}

} // namespace xinfer::zoo::dsp