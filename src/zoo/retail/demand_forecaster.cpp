#include <include/zoo/retail/demand_forecaster.h>
#include <stdexcept>

namespace xinfer::zoo::retail {

    struct DemandForecaster::Impl {
        DemandForecasterConfig config_;
        // This class is a specialized wrapper around the generic timeseries::Forecaster
        std::unique_ptr<timeseries::Forecaster> forecaster_;
    };

    DemandForecaster::DemandForecaster(const DemandForecasterConfig& config)
        : pimpl_(new Impl{config})
    {
        timeseries::ForecasterConfig ts_config;
        ts_config.engine_path = pimpl_->config_.engine_path;
        ts_config.input_sequence_length = pimpl_->config_.input_sequence_length;
        ts_config.output_sequence_length = pimpl_->config_.output_sequence_length;

        pimpl_->forecaster_ = std::make_unique<timeseries::Forecaster>(ts_config);
    }

    DemandForecaster::~DemandForecaster() = default;
    DemandForecaster::DemandForecaster(DemandForecaster&&) noexcept = default;
    DemandForecaster& DemandForecaster::operator=(DemandForecaster&&) noexcept = default;

    std::vector<float> DemandForecaster::predict(const std::vector<float>& historical_sales) {
        if (!pimpl_) throw std::runtime_error("DemandForecaster is in a moved-from state.");

        // A more complex implementation could add retail-specific features here,
        // like holidays or promotional events, before passing to the generic forecaster.

        return pimpl_->forecaster_->predict(historical_sales);
    }

} // namespace xinfer::zoo::retail