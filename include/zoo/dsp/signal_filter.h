#pragma once

#include <vector>
#include <memory>
#include <string>

namespace xinfer::zoo::dsp {

    enum class FilterType {
        LOW_PASS = 0,
        HIGH_PASS = 1,
        BAND_PASS = 2,
        NOTCH = 3
    };

    struct FilterConfig {
        FilterType type;
        double sample_rate_hz;

        // Cutoff frequency (Hz)
        double cutoff_freq_hz;

        // For Band-pass/Notch, a second frequency is needed
        double cutoff_freq2_hz = 0;

        // Filter order (higher order = sharper cutoff)
        int order = 4;
    };

    class SignalFilter {
    public:
        explicit SignalFilter(const FilterConfig& config);
        ~SignalFilter();

        // Move semantics
        SignalFilter(SignalFilter&&) noexcept;
        SignalFilter& operator=(SignalFilter&&) noexcept;
        SignalFilter(const SignalFilter&) = delete;
        SignalFilter& operator=(const SignalFilter&) = delete;

        /**
         * @brief Apply the filter to a signal.
         *
         * @param input_signal Raw 1D signal (e.g., from a sensor).
         * @return The filtered signal.
         */
        std::vector<float> apply(const std::vector<float>& input_signal);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::dsp