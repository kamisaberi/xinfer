#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::dsp {

    enum class FilterType {
        LOW_PASS,
        HIGH_PASS,
        BAND_PASS
    };

    struct SignalFilterConfig {
        FilterType type;
        int sample_rate = 44100;
        float cutoff_freq1 = 1000.0f;
        float cutoff_freq2 = 0.0f; // Used for band-pass
        int filter_length = 129; // Should be odd
    };

    class SignalFilter {
    public:
        explicit SignalFilter(const SignalFilterConfig& config);
        ~SignalFilter();

        SignalFilter(const SignalFilter&) = delete;
        SignalFilter& operator=(const SignalFilter&) = delete;
        SignalFilter(SignalFilter&&) noexcept;
        SignalFilter& operator=(SignalFilter&&) noexcept;

        std::vector<float> process(const std::vector<float>& input_signal);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::dsp

