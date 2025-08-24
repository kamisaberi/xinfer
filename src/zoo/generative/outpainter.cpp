#include <include/zoo/generative/outpainter.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct Outpainter::Impl {
    OutpainterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_image_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_mask_;
};

Outpainter::Outpainter(const OutpainterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Outpainting engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Outpainting engine must have exactly two inputs (image and mask).");
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

Outpainter::~Outpainter() = default;
Outpainter::Outpainter(Outpainter&&) noexcept = default;
Outpainter& Outpainter::operator=(Outpainter&&) noexcept = default;

cv::Mat Outpainter::predict(const cv::Mat& image, int top, int right, int bottom, int left) {
    if (!pimpl_) throw std::runtime_error("Outpainter is in a moved-from state.");

    int new_width = image.cols + left + right;
    int new_height = image.rows + top + bottom;

    cv::Mat expanded_image = cv::Mat::zeros(new_height, new_width, image.type());
    cv::Rect roi(left, top, image.cols, image.rows);
    image.copyTo(expanded_image(roi));

    cv::Mat mask = cv::Mat::zeros(new_height, new_width, CV_8UC1);
    cv::rectangle(mask, roi, cv::Scalar(255), -1);
    cv::bitwise_not(mask, mask); // Invert mask for outpainting

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor image_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_image_->process(expanded_image, image_tensor);

    auto mask_input_shape = pimpl_->engine_->get_input_shape(1);
    core::Tensor mask_tensor(mask_input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_mask_->process(mask, mask_tensor);

    auto output_tensors = pimpl_->engine_->infer({image_tensor, mask_tensor});
    const core::Tensor& outpainted_image_tensor = output_tensors[0];

    auto output_shape = outpainted_image_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(outpainted_image_tensor.num_elements());
    outpainted_image_tensor.copy_to_host(h_output.data());

    cv::Mat outpainted_image_chw;
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, outpainted_image_chw);

    cv::Mat outpainted_image_bgr, final_image;
    outpainted_image_chw.convertTo(outpainted_image_bgr, CV_8UC3, 255.0, 127.5);

    cv::resize(outpainted_image_bgr, final_image, cv::Size(new_width, new_height), 0, 0, cv::INTER_CUBIC);

    return final_image;
}

} // namespace xinfer::zoo::generative