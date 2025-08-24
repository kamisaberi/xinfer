#include <include/zoo/vision/low_light_enhancer.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct LowLightEnhancer::Impl {
    LowLightEnhancerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

LowLightEnhancer::LowLightEnhancer(const LowLightEnhancerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Low light enhancement engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Typically these models work on [0,1] scaled images, no mean/std normalization
        std::vector<float>{0.0f, 0.0f, 0.0f},
        std::vector<float>{1.0f, 1.0f, 1.0f}
    );
}

LowLightEnhancer::~LowLightEnhancer() = default;
LowLightEnhancer::LowLightEnhancer(LowLightEnhancer&&) noexcept = default;
LowLightEnhancer& LowLightEnhancer::operator=(LowLightEnhancer&&) noexcept = default;

cv::Mat LowLightEnhancer::predict(const cv::Mat& low_light_image) {
    if (!pimpl_) throw std::runtime_error("LowLightEnhancer is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(low_light_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& enhanced_image_tensor = output_tensors[0];

    auto output_shape = enhanced_image_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(enhanced_image_tensor.num_elements());
    enhanced_image_tensor.copy_to_host(h_output.data());

    cv::Mat enhanced_image_chw(H, W, CV_32FC3);
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, enhanced_image_chw);

    cv::Mat enhanced_image_bgr, final_image;
    enhanced_image_chw.convertTo(enhanced_image_bgr, CV_8UC3, 255.0);

    cv::resize(enhanced_image_bgr, final_image, low_light_image.size(), 0, 0, cv::INTER_CUBIC);

    return final_image;
}

} // namespace xinfer::zoo::vision