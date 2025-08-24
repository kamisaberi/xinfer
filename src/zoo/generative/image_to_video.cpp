#include <include/zoo/generative/image_to_video.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct ImageToVideo::Impl {
    ImageToVideoConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

ImageToVideo::ImageToVideo(const ImageToVideoConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("ImageToVideo engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f}
    );
}

ImageToVideo::~ImageToVideo() = default;
ImageToVideo::ImageToVideo(ImageToVideo&&) noexcept = default;
ImageToVideo& ImageToVideo::operator=(ImageToVideo&&) noexcept = default;

std::vector<cv::Mat> ImageToVideo::predict(const cv::Mat& start_image) {
    if (!pimpl_) throw std::runtime_error("ImageToVideo is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(start_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& video_tensor = output_tensors[0];

    auto output_shape = video_tensor.shape();
    const int F = output_shape[1];
    const int C = output_shape[2];
    const int H = output_shape[3];
    const int W = output_shape[4];

    std::vector<float> h_output(video_tensor.num_elements());
    video_tensor.copy_to_host(h_output.data());

    std::vector<cv::Mat> frames;
    for (int f = 0; f < F; ++f) {
        cv::Mat frame_chw(H, W, CV_32FC3);
        std::vector<cv::Mat> channels;
        for(int c = 0; c < C; ++c) {
            channels.emplace_back(H, W, CV_32F, h_output.data() + f * C * H * W + c * H * W);
        }
        cv::merge(channels, frame_chw);

        cv::Mat frame_bgr, final_frame;
        frame_chw.convertTo(frame_bgr, CV_8UC3, 255.0);

        cv::resize(frame_bgr, final_frame, start_image.size(), 0, 0, cv::INTER_CUBIC);
        frames.push_back(final_frame);
    }

    return frames;
}

} // namespace xinfer::zoo::generative