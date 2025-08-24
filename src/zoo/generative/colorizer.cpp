#include <include/zoo/generative/colorizer.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct Colorizer::Impl {
    ColorizerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

Colorizer::Colorizer(const ColorizerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Colorizer engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.5f},
        std::vector<float>{0.5f}
    );
}

Colorizer::~Colorizer() = default;
Colorizer::Colorizer(Colorizer&&) noexcept = default;
Colorizer& Colorizer::operator=(Colorizer&&) noexcept = default;

cv::Mat Colorizer::predict(const cv::Mat& grayscale_image) {
    if (!pimpl_) throw std::runtime_error("Colorizer is in a moved-from state.");

    cv::Mat input_gray;
    if (grayscale_image.channels() == 3) {
        cv::cvtColor(grayscale_image, input_gray, cv::COLOR_BGR2GRAY);
    } else {
        input_gray = grayscale_image;
    }

    cv::Mat lab_image;
    input_gray.convertTo(lab_image, CV_32F, 1.0 / 255.0);
    cv::cvtColor(lab_image, lab_image, cv::COLOR_GRAY2BGR);
    cv::cvtColor(lab_image, lab_image, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> lab_channels(3);
    cv::split(lab_image, lab_channels);
    cv::Mat l_channel = lab_channels[0];

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(l_channel, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& ab_tensor = output_tensors[0];

    auto output_shape = ab_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_ab(ab_tensor.num_elements());
    ab_tensor.copy_to_host(h_ab.data());

    cv::Mat a_channel(H, W, CV_32F, h_ab.data());
    cv::Mat b_channel(H, W, CV_32F, h_ab.data() + H * W);

    cv::Mat resized_l, resized_a, resized_b;
    cv::resize(l_channel, resized_l, cv::Size(image.cols, image.rows));
    cv::resize(a_channel, resized_a, cv::Size(image.cols, image.rows));
    cv::resize(b_channel, resized_b, cv::Size(image.cols, image.rows));

    cv::Mat final_lab;
    cv::merge(std::vector<cv::Mat>{resized_l, resized_a, resized_b}, final_lab);

    cv::Mat final_bgr;
    cv::cvtColor(final_lab, final_bgr, cv::COLOR_Lab2BGR);

    final_bgr.convertTo(final_bgr, CV_8UC3, 255.0);

    return final_bgr;
}

} // namespace xinfer::zoo::generative