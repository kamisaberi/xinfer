#pragma once


#include <string>
#include <vector>
#include <memory>

#include <include/zoo/timeseries/forecaster.h>

namespace xinfer::zoo::retail {

    struct DemandForecasterConfig {
        std::string engine_path;
        int input_sequence_length = 90; // e.g., 90 days of history
        int output_sequence_length = 14; // e.g., predict next 14 days
    };

    class DemandForecaster {
    public:
        explicit DemandForecaster(const DemandForecasterConfig& config);
        ~DemandForecaster();

        DemandForecaster(const DemandForecaster&) = delete;
        DemandForecaster& operator=(const DemandForecaster&) = delete;
        DemandForecaster(DemandForecaster&&) noexcept;
        DemandForecaster& operator=(DemandForecaster&&) noexcept;

        std::vector<float> predict(const std::vector<float>& historical_sales);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail

