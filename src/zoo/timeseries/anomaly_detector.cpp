#include <include/zoo/timeseries/anomaly_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>

#include <include/core/engine.h>
#include <include/core/tensor.h>

namespace xinfer::zoo::timeseries {

struct AnomalyDetector::Impl {
    AnomalyDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
};

AnomalyDetector::AnomalyDetector(const AnomalyDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Time-series anomaly detector engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
}

AnomalyDetector::~AnomalyDetector() = default;
AnomalyDetector::AnomalyDetector(AnomalyDetector&&) noexcept = default;
AnomalyDetector& AnomalyDetector::operator=(AnomalyDetector&&) noexcept = default;

AnomalyResult AnomalyDetector::predict(const std::vector<float>& time_series_window) {
    if (!pimpl_) throw std::runtime_error("AnomalyDetector is in a moved-from state.");
    if (time_series_window.size() != pimpl_->config_.sequence_length) {
        throw std::invalid_argument("Input time_series_window size does not match model's expected sequence_length.");
    }

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    input_tensor.copy_from_host(time_series_window.data());

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& reconstructed_tensor = output_tensors[0];

    std::vector<float> reconstructed_window(reconstructed_tensor.num_elements());
    reconstructed_tensor.copy_to_host(reconstructed_window.data());

    AnomalyResult result;
    result.reconstruction_error.resize(pimpl_->config_.sequence_length);
    double total_squared_error = 0.0;

    for (int i = 0; i < pimpl_->config_.sequence_length; ++i) {
        double diff = time_series_window[i] - reconstructed_window[i];
        result.reconstruction_error[i] = diff * diff;
        total_squared_error += result.reconstruction_error[i];
    }

    result.anomaly_score = total_squared_error / pimpl_->config_.sequence_length;
    result.is_anomaly = result.anomaly_score > pimpl_->config_.anomaly_threshold;

    return result;
}

} // namespace xinfer::zoo::timeseries