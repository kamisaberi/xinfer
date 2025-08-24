#include <include/zoo/dsp/signal_filter.h>
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

#define CHECK_CUDA(call) { const cudaError_t e = call; if (e != cudaSuccess) { throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(e))); } }
#define CHECK_CUFFT(call) { const cufftResult_t s = call; if (s != CUFFT_SUCCESS) { throw std::runtime_error("cuFFT Error"); } }

namespace xinfer::zoo::dsp {

__global__ void complex_multiply_kernel(cufftComplex* a, const cufftComplex* b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        float real_a = a[i].x;
        float imag_a = a[i].y;
        float real_b = b[i].x;
        float imag_b = b[i].y;
        a[i].x = real_a * real_b - imag_a * imag_b;
        a[i].y = real_a * imag_b + imag_a * real_b;
    }
}

__global__ void normalize_kernel(float* signal, float norm_factor, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        signal[i] *= norm_factor;
    }
}

struct SignalFilter::Impl {
    SignalFilterConfig config_;
    cufftHandle fft_plan_;
    int fft_size_;
    float* d_kernel_ = nullptr;
    cufftComplex* d_kernel_fft_ = nullptr;

    Impl(const SignalFilterConfig& config) : config_(config) {
        if (config.filter_length % 2 == 0) {
            throw std::invalid_argument("Filter length must be odd.");
        }
    }

    ~Impl() {
        cudaFree(d_kernel_);
        cudaFree(d_kernel_fft_);
        if (fft_plan_) {
            cufftDestroy(fft_plan_);
        }
    }
};

SignalFilter::SignalFilter(const SignalFilterConfig& config) : pimpl_(new Impl(config)) {}
SignalFilter::~SignalFilter() = default;
SignalFilter::SignalFilter(SignalFilter&&) noexcept = default;
SignalFilter& SignalFilter::operator=(SignalFilter&&) noexcept = default;

std::vector<float> SignalFilter::process(const std::vector<float>& input_signal) {
    if (!pimpl_) throw std::runtime_error("SignalFilter is in a moved-from state.");

    const int signal_len = input_signal.size();
    const int kernel_len = pimpl_->config_.filter_length;
    pimpl_->fft_size_ = 1;
    while (pimpl_->fft_size_ < (signal_len + kernel_len - 1)) {
        pimpl_->fft_size_ *= 2;
    }

    CHECK_CUFFT(cufftPlan1d(&pimpl_->fft_plan_, pimpl_->fft_size_, CUFFT_R2C, 1));

    std::vector<float> h_kernel(kernel_len, 0.0f);
    float fc1 = pimpl_->config_.cutoff_freq1 / pimpl_->config_.sample_rate;
    float fc2 = pimpl_->config_.cutoff_freq2 / pimpl_->config_.sample_rate;
    int center = kernel_len / 2;
    for (int i = 0; i < kernel_len; ++i) {
        int n = i - center;
        if (n == 0) {
            h_kernel[i] = 2.0f * M_PI * fc1;
        } else {
            h_kernel[i] = sinf(2.0f * M_PI * fc1 * n) / n;
        }
        h_kernel[i] *= (0.54f - 0.46f * cosf(2.0f * M_PI * i / (kernel_len - 1)));
    }

    if (pimpl_->config_.type == FilterType::HIGH_PASS) {
        for (int i = 0; i < kernel_len; ++i) h_kernel[i] = -h_kernel[i];
        h_kernel[center] += 1.0f;
    }

    CHECK_CUDA(cudaMalloc(&pimpl_->d_kernel_, pimpl_->fft_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemset(pimpl_->d_kernel_, 0, pimpl_->fft_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(pimpl_->d_kernel_, h_kernel.data(), kernel_len * sizeof(float), cudaMemcpyHostToDevice));

    const int fft_complex_size = (pimpl_->fft_size_ / 2) + 1;
    CHECK_CUDA(cudaMalloc(&pimpl_->d_kernel_fft_, fft_complex_size * sizeof(cufftComplex)));
    CHECK_CUFFT(cufftExecR2C(pimpl_->fft_plan_, pimpl_->d_kernel_, pimpl_->d_kernel_fft_));

    float* d_signal;
    CHECK_CUDA(cudaMalloc(&d_signal, pimpl_->fft_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemset(d_signal, 0, pimpl_->fft_size_ * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_signal, input_signal.data(), signal_len * sizeof(float), cudaMemcpyHostToDevice));

    cufftComplex* d_signal_fft;
    CHECK_CUDA(cudaMalloc(&d_signal_fft, fft_complex_size * sizeof(cufftComplex)));
    CHECK_CUFFT(cufftExecR2C(pimpl_->fft_plan_, d_signal, d_signal_fft));

    int threads = 256;
    int blocks = (fft_complex_size + threads - 1) / threads;
    complex_multiply_kernel<<<blocks, threads>>>(d_signal_fft, pimpl_->d_kernel_fft_, fft_complex_size);

    cufftHandle ifft_plan;
    CHECK_CUFFT(cufftPlan1d(&ifft_plan, pimpl_->fft_size_, CUFFT_C2R, 1));
    CHECK_CUFFT(cufftExecC2R(ifft_plan, d_signal_fft, d_signal));

    normalize_kernel<<< (pimpl_->fft_size_ + threads - 1) / threads, threads >>>(d_signal, 1.0f / pimpl_->fft_size_, pimpl_->fft_size_);

    std::vector<float> filtered_signal(signal_len);
    CHECK_CUDA(cudaMemcpy(filtered_signal.data(), d_signal + center, signal_len * sizeof(float), cudaMemcpyDeviceToHost));

    cudaFree(d_signal);
    cudaFree(d_signal_fft);
    cufftDestroy(ifft_plan);

    return filtered_signal;
}

} // namespace xinfer::zoo::dsp