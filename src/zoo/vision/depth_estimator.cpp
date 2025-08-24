#include <include/zoo/vision/depth_estimator.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct DepthEstimator::Impl {
    DepthEstimatorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

DepthEstimator::DepthEstimator(const DepthEstimatorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Depth estimation engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // MiDaS models often use simple [0, 1] normalization
        std::vector<float>{0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f}
    );
}

DepthEstimator::~DepthEstimator() = default;
DepthEstimator::DepthEstimator(DepthEstimator&&) noexcept = default;
DepthEstimator& DepthEstimator::operator=(DepthEstimator&&) noexcept = default;

cv::Mat DepthEstimator::predict(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DepthEstimator is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& inverse_depth_tensor = output_tensors[0];

    auto output_shape = inverse_depth_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    cv::Mat inverse_depth_map(H, W, CV_32F);
    inverse_depth_tensor.copy_to_host(inverse_depth_map.data);

    double min_val, max_val;
    cv::minMaxLoc(inverse_depth_map, &min_val, &max_val);

    cv::Mat normalized_depth_map;
    if (max_val - min_val > 0) {
        inverse_depth_map.convertTo(normalized_depth_map, CV_32F, 1.0 / (max_val - min_val), -min_val / (max_val - min_val));
    } else {
        normalized_depth_map = cv::Mat::zeros(H, W, CV_32F);
    }

    cv::Mat final_map;
    cv::resize(normalized_depth_map, final_map, image.size(), 0, 0, cv::INTER_CUBIC);

    return final_map;
}

} // namespace xinfer::zoo::vision