#include <include/zoo/geospatial/crop_monitor.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::geospatial {

struct CropMonitor::Impl {
    CropMonitorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

CropMonitor::CropMonitor(const CropMonitorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Crop monitor engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Normalization for multi-spectral images can vary greatly
        std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f, 0.5f}
    );
}

CropMonitor::~CropMonitor() = default;
CropMonitor::CropMonitor(CropMonitor&&) noexcept = default;
CropMonitor& CropMonitor::operator=(CropMonitor&&) noexcept = default;

cv::Mat CropMonitor::predict_health_map(const cv::Mat& multispectral_image) {
    if (!pimpl_) throw std::runtime_error("CropMonitor is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);

    // The preprocessor needs to handle multi-channel inputs beyond 3
    // pimpl_->preprocessor_->process(multispectral_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& health_map_tensor = output_tensors[0];

    auto output_shape = health_map_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    cv::Mat health_map(H, W, CV_32F);
    health_map_tensor.copy_to_host(health_map.data);

    cv::Mat final_map;
    cv::resize(health_map, final_map, multispectral_image.size(), 0, 0, cv::INTER_CUBIC);

    return final_map;
}

cv::Mat CropMonitor::calculate_ndvi(const cv::Mat& multispectral_image) {
    if (multispectral_image.channels() < 4) {
        throw std::invalid_argument("NDVI calculation requires at least 4 channels (e.g., Blue, Green, Red, NIR).");
    }

    std::vector<cv::Mat> bands;
    cv::split(multispectral_image, bands);

    cv::Mat red, nir;
    // Assuming B, G, R, NIR order
    bands[2].convertTo(red, CV_32F);
    bands[3].convertTo(nir, CV_32F);

    cv::Mat ndvi = (nir - red) / (nir + red + 1e-10); // Add epsilon to avoid division by zero

    return ndvi;
}

} // namespace xinfer::zoo::geospatial