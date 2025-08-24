#include <include/zoo/threed/slam_accelerator.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::threed {

struct SLAMAccelerator::Impl {
    SLAMAcceleratorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

SLAMAccelerator::SLAMAccelerator(const SLAMAcceleratorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.feature_engine_path).good()) {
        throw std::runtime_error("SLAM feature engine file not found: " + pimpl_->config_.feature_engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.feature_engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Typically, feature extractors use simple normalization
        std::vector<float>{0.0f},
        std::vector<float>{1.0f}
    );
}

SLAMAccelerator::~SLAMAccelerator() = default;
SLAMAccelerator::SLAMAccelerator(SLAMAccelerator&&) noexcept = default;
SLAMAccelerator& SLAMAccelerator::operator=(SLAMAccelerator&&) noexcept = default;

SLAMFeatureResult SLAMAccelerator::extract_features(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SLAMAccelerator is in a moved-from state.");

    cv::Mat gray_image;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = image;
    }

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(gray_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});

    // Assumes a model like SuperPoint with two outputs:
    // 1. A heatmap of keypoint locations
    // 2. A dense tensor of descriptors
    const core::Tensor& keypoints_tensor = output_tensors[0];
    const core::Tensor& descriptors_tensor = output_tensors[1];

    SLAMFeatureResult result;

    // A full implementation would have a dedicated post-processing CUDA kernel
    // to perform NMS on the heatmap and gather the corresponding descriptors.
    // This is a placeholder for that complex logic.

    return result;
}

} // namespace xinfer::zoo::threed