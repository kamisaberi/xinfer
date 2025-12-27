#include <xinfer/zoo/dsp/signal_filter.h>
#include <xinfer/core/logging.h>

#include <opencv2/opencv.hpp>
#include <cmath>
#include <numeric>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xinfer::zoo::dsp {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SignalFilter::Impl {
    FilterConfig config_;

    // The FIR filter coefficients (the kernel)
    cv::Mat kernel_;

    Impl(const FilterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // --- Design the FIR Filter Kernel ---
        // This is a simplified FIR design.
        // For production, use a library like liquid-dsp or SciPy (via C++)
        // to design more complex IIR or Butterworth filters.

        int N = config_.order * 2 + 1; // Kernel size
        kernel_ = cv::Mat(1, N, CV_32F);

        // Normalized cutoff frequency (0.0 to 0.5)
        float fc = (float)(config_.cutoff_freq_hz / config_.sample_rate_hz);

        // Window function (Hann)
        std::vector<float> window(N);
        for(int i=0; i<N; ++i) {
            window[i] = 0.5f * (1.0f - std::cos(2.0 * M_PI * i / (N - 1)));
        }

        // Sinc function
        for (int i = 0; i < N; ++i) {
            float n = (float)i - (N - 1) / 2.0f;
            float sinc = (n == 0) ? 1.0f : std::sin(2 * M_PI * fc * n) / (M_PI * n);

            kernel_.at<float>(0, i) = 2 * fc * sinc * window[i];
        }

        // Normalize kernel to have gain of 1 at DC
        float sum = (float)cv::sum(kernel_)[0];
        if (sum > 1e-6) {
            kernel_ /= sum;
        }

        // --- Spectral Inversion for High-Pass ---
        if (config_.type == FilterType::HIGH_PASS) {
            // HPF = All-Pass - LPF
            // All-pass is an impulse at center
            cv::Mat impulse = cv::Mat::zeros(1, N, CV_32F);
            impulse.at<float>(0, N/2) = 1.0f;

            kernel_ = impulse - kernel_;
        }
        // Note: Band-pass/Notch are combinations of LPF/HPF and are more complex. Omitted here.
    }
};

// =================================================================================
// Public API
// =================================================================================

SignalFilter::SignalFilter(const FilterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SignalFilter::~SignalFilter() = default;
SignalFilter::SignalFilter(SignalFilter&&) noexcept = default;
SignalFilter& SignalFilter::operator=(SignalFilter&&) noexcept = default;

std::vector<float> SignalFilter::apply(const std::vector<float>& input_signal) {
    if (!pimpl_) throw std::runtime_error("SignalFilter is null.");
    if (input_signal.empty()) return {};

    // 1. Wrap input in cv::Mat
    // Note: This makes a copy. For zero-copy, pass raw pointers.
    cv::Mat input_mat(input_signal, true);
    input_mat = input_mat.reshape(1, 1); // Treat as 1-row image

    // 2. Apply Convolution (Filter)
    cv::Mat output_mat;
    // 'filter2D' is a fast, SIMD-optimized 1D/2D convolution
    cv::filter2D(input_mat, output_mat, -1, pimpl_->kernel_, cv::Point(-1,-1), 0, cv::BORDER_REPLICATE);

    // 3. Convert back to vector
    std::vector<float> output_signal;
    if (output_mat.isContinuous()) {
        output_signal.assign((float*)output_mat.datastart, (float*)output_mat.dataend);
    } else {
        // Slower path for non-continuous mats
        for (int i = 0; i < output_mat.rows; ++i) {
            output_signal.insert(output_signal.end(), output_mat.ptr<float>(i), output_mat.ptr<float>(i) + output_mat.cols);
        }
    }

    return output_signal;
}

} // namespace xinfer::zoo::dsp