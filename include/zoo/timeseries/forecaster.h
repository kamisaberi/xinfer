#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::zoo::timeseries {

    struct ForecasterConfig {
        std::string engine_path;
        int input_sequence_length = 128;
        int output_sequence_length = 32;
    };

    class Forecaster {
    public:
        explicit Forecaster(const ForecasterConfig& config);
        ~Forecaster();

        Forecaster(const Forecaster&) = delete;
        Forecaster& operator=(const Forecaster&) = delete;
        Forecaster(Forecaster&&) noexcept;
        Forecaster& operator=(Forecaster&&) noexcept;

        std::vector<float> predict(const std::vector<float>& historical_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::timeseries

