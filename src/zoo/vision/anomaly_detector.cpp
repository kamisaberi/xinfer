#include <include/zoo/vision/anomaly_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/anomaly.h> // Assumed to exist for this implementation

namespace xinfer::zoo::vision {

struct AnomalyDetector::Impl {
    AnomalyDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

AnomalyDetector::AnomalyDetector(const AnomalyDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Anomaly detection engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        pimpl_->config_.mean,
        pimpl_->config_.std
    );
}

AnomalyDetector::~AnomalyDetector() = default;
AnomalyDetector::AnomalyDetector(AnomalyDetector&&) noexcept = default;
AnomalyDetector& AnomalyDetector::operator=(AnomalyDetector&&) noexcept = default;

AnomalyResult AnomalyDetector::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("AnomalyDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& reconstructed_tensor = output_tensors[0];

    core::Tensor anomaly_map_tensor;
    float anomaly_score;

    postproc::anomaly::calculate_reconstruction_error(
        input_tensor,
        reconstructed_tensor,
        anomaly_map_tensor,
        anomaly_score
    );

    cv::Mat anomaly_map(input_shape[2], input_shape[3], CV_32F);
    anomaly_map_tensor.copy_to_host(anomaly_map.data);

    AnomalyResult result;
    result.anomaly_score = anomaly_score;
    result.is_anomaly = anomaly_score > pimpl_->config_.anomaly_threshold;
    result.anomaly_map = anomaly_map;

    return result;
}

} // namespace xinfer::zoo::vision