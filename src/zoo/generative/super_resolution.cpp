#include <include/zoo/generative/super_resolution.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct SuperResolution::Impl {
    SuperResolutionConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

SuperResolution::SuperResolution(const SuperResolutionConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Super resolution engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // ESRGAN models typically expect [0,1] range
        std::vector<float>{0.0f, 0.0f, 0.0f},
        std::vector<float>{1.0f, 1.0f, 1.0f}
    );
}

SuperResolution::~SuperResolution() = default;
SuperResolution::SuperResolution(SuperResolution&&) noexcept = default;
SuperResolution& SuperResolution::operator=(SuperResolution&&) noexcept = default;

cv::Mat SuperResolution::predict(const cv::Mat& low_res_image) {
    if (!pimpl_) throw std::runtime_error("SuperResolution is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(low_res_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& high_res_tensor = output_tensors[0];

    auto output_shape = high_res_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(high_res_tensor.num_elements());
    high_res_tensor.copy_to_host(h_output.data());

    cv::Mat high_res_image_chw;
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, high_res_image_chw);

    cv::Mat high_res_image_bgr;
    high_res_image_chw.convertTo(high_res_image_bgr, CV_8UC3, 255.0);

    cv::Mat final_image;
    if (low_res_image.size() == cv::Size(W, H)) {
        final_image = high_res_image_bgr;
    } else {
        cv::resize(high_res_image_bgr, final_image,
                   cv::Size(low_res_image.cols * pimpl_->config_.upscale_factor, low_res_image.rows * pimpl_->config_.upscale_factor),
                   0, 0, cv::INTER_CUBIC);
    }

    return final_image;
}

} // namespace xinfer::zoo::generative