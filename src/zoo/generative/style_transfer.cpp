#include <include/zoo/generative/style_transfer.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct StyleTransfer::Impl {
    StyleTransferConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

StyleTransfer::StyleTransfer(const StyleTransferConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Style transfer engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

StyleTransfer::~StyleTransfer() = default;
StyleTransfer::StyleTransfer(StyleTransfer&&) noexcept = default;
StyleTransfer& StyleTransfer::operator=(StyleTransfer&&) noexcept = default;

cv::Mat StyleTransfer::predict(const cv::Mat& content_image) {
    if (!pimpl_) throw std::runtime_error("StyleTransfer is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(content_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& styled_image_tensor = output_tensors[0];

    auto output_shape = styled_image_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(styled_image_tensor.num_elements());
    styled_image_tensor.copy_to_host(h_output.data());

    cv::Mat styled_image_chw;
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, styled_image_chw);

    // De-normalize the image from ImageNet stats
    cv::Mat mean(H, W, CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std(H, W, CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    cv::multiply(styled_image_chw, std, styled_image_chw);
    cv::add(styled_image_chw, mean, styled_image_chw);

    cv::Mat styled_image_bgr, final_image;
    styled_image_chw.convertTo(styled_image_bgr, CV_8UC3, 255.0);

    cv::resize(styled_image_bgr, final_image, content_image.size(), 0, 0, cv::INTER_CUBIC);

    return final_image;
}

} // namespace xinfer::zoo::generative