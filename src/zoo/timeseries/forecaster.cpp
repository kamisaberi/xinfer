#include <include/zoo/timeseries/forecaster.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/core/tensor.h>

namespace xinfer::zoo::timeseries {

    struct Forecaster::Impl {
        ForecasterConfig config_;
        std::unique_ptr<core::InferenceEngine> engine_;
    };

    Forecaster::Forecaster(const ForecasterConfig& config)
        : pimpl_(new Impl{config})
    {
        if (!std::ifstream(pimpl_->config_.engine_path).good()) {
            throw std::runtime_error("Time-series forecaster engine file not found: " + pimpl_->config_.engine_path);
        }

        pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    }

    Forecaster::~Forecaster() = default;
    Forecaster::Forecaster(Forecaster&&) noexcept = default;
    Forecaster& Forecaster::operator=(Forecaster&&) noexcept = default;

    std::vector<float> Forecaster::predict(const std::vector<float>& historical_data) {
        if (!pimpl_) throw std::runtime_error("Forecaster is in a moved-from state.");
        if (historical_data.size() != pimpl_->config_.input_sequence_length) {
            throw std::invalid_argument("Input historical_data size does not match model's expected input_sequence_length.");
        }

        auto input_shape = pimpl_->engine_->get_input_shape(0);
        core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
        input_tensor.copy_from_host(historical_data.data());

        auto output_tensors = pimpl_->engine_->infer({input_tensor});
        const core::Tensor& forecast_tensor = output_tensors[0];

        std::vector<float> forecast(forecast_tensor.num_elements());
        forecast_tensor.copy_to_host(forecast.data());

        if (forecast.size() != pimpl_->config_.output_sequence_length) {
            // This is a sanity check. The engine's output should match the config.
        }

        return forecast;
    }

} // namespace xinfer::zoo::timeseries