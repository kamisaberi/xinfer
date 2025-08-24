#include <include/zoo/vision/image_deblur.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct ImageDeblur::Impl {
    ImageDeblurConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

ImageDeblur::ImageDeblur(const ImageDeblurConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Image deblur engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f}
    );
}

ImageDeblur::~ImageDeblur() = default;
ImageDeblur::ImageDeblur(ImageDeblur&&) noexcept = default;
ImageDeblur& ImageDeblur::operator=(ImageDeblur&&) noexcept = default;

cv::Mat ImageDeblur::predict(const cv::Mat& blurry_image) {
    if (!pimpl_) throw std::runtime_error("ImageDeblur is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(blurry_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& sharp_image_tensor = output_tensors[0];

    auto output_shape = sharp_image_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(sharp_image_tensor.num_elements());
    sharp_image_tensor.copy_to_host(h_output.data());

    cv::Mat sharp_image_chw(H, W, CV_32FC3);

    for (int c = 0; c < C; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float val = h_output[c * H * W + y * W + x];
                sharp_image_chw.at<cv::Vec3f>(y, x)[c] = val;
            }
        }
    }

    cv::Mat sharp_image_bgr, final_image;
    sharp_image_chw.convertTo(sharp_image_bgr, CV_8UC3, 255.0, -127.5);

    cv::resize(sharp_image_bgr, final_image, blurry_image.size(), 0, 0, cv::INTER_CUBIC);

    return final_image;
}

} // namespace xinfer::zoo::vision