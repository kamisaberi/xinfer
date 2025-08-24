#include <include/zoo/generative/inpainter.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct Inpainter::Impl {
    InpainterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_image_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_mask_;
};

Inpainter::Inpainter(const InpainterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Inpainting engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Inpainting engine must have exactly two inputs (image and mask).");
    }

    pimpl_->preprocessor_image_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f}
    );

    pimpl_->preprocessor_mask_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.0f},
        std::vector<float>{1.0f}
    );
}

Inpainter::~Inpainter() = default;
Inpainter::Inpainter(Inpainter&&) noexcept = default;
Inpainter& Inpainter::operator=(Inpainter&&) noexcept = default;

cv::Mat Inpainter::predict(const cv::Mat& image, const cv::Mat& mask) {
    if (!pimpl_) throw std::runtime_error("Inpainter is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor image_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_image_->process(image, image_tensor);

    auto mask_input_shape = pimpl_->engine_->get_input_shape(1);
    core::Tensor mask_tensor(mask_input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_mask_->process(mask, mask_tensor);

    auto output_tensors = pimpl_->engine_->infer({image_tensor, mask_tensor});
    const core::Tensor& inpainted_image_tensor = output_tensors[0];

    auto output_shape = inpainted_image_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(inpainted_image_tensor.num_elements());
    inpainted_image_tensor.copy_to_host(h_output.data());

    cv::Mat inpainted_image_chw;
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, inpainted_image_chw);

    cv::Mat inpainted_image_bgr, final_image;
    inpainted_image_chw.convertTo(inpainted_image_bgr, CV_8UC3, 255.0, 127.5);

    cv::resize(inpainted_image_bgr, final_image, image.size(), 0, 0, cv::INTER_CUBIC);

    cv::Mat mask_resized;
    cv::resize(mask, mask_resized, image.size());
    cv::cvtColor(mask_resized, mask_resized, cv::COLOR_GRAY2BGR);

    cv::Mat original_image_uchar;
    image.convertTo(original_image_uchar, CV_8UC3);

    final_image.copyTo(original_image_uchar, mask_resized);

    return original_image_uchar;
}

} // namespace xinfer::zoo::generative