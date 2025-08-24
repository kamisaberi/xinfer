#include <include/zoo/generative/video_frame_interpolation.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::generative {

struct VideoFrameInterpolation::Impl {
    VideoFrameInterpolationConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

VideoFrameInterpolation::VideoFrameInterpolation(const VideoFrameInterpolationConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Video frame interpolation engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Video frame interpolation engine must have exactly two inputs.");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Models often expect [0,1] range
        std::vector<float>{0.0f, 0.0f, 0.0f},
        std::vector<float>{1.0f, 1.0f, 1.0f}
    );
}

VideoFrameInterpolation::~VideoFrameInterpolation() = default;
VideoFrameInterpolation::VideoFrameInterpolation(VideoFrameInterpolation&&) noexcept = default;
VideoFrameInterpolation& VideoFrameInterpolation::operator=(VideoFrameInterpolation&&) noexcept = default;

cv::Mat VideoFrameInterpolation::predict(const cv::Mat& frame_before, const cv::Mat& frame_after) {
    if (!pimpl_) throw std::runtime_error("VideoFrameInterpolation is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor_before(input_shape, core::DataType::kFLOAT);
    core::Tensor input_tensor_after(input_shape, core::DataType::kFLOAT);

    pimpl_->preprocessor_->process(frame_before, input_tensor_before);
    pimpl_->preprocessor_->process(frame_after, input_tensor_after);

    auto output_tensors = pimpl_->engine_->infer({input_tensor_before, input_tensor_after});
    const core::Tensor& interpolated_frame_tensor = output_tensors[0];

    auto output_shape = interpolated_frame_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> h_output(interpolated_frame_tensor.num_elements());
    interpolated_frame_tensor.copy_to_host(h_output.data());

    cv::Mat interpolated_chw;
    std::vector<cv::Mat> channels;
    for(int i = 0; i < C; ++i) {
        channels.emplace_back(H, W, CV_32F, h_output.data() + i * H * W);
    }
    cv::merge(channels, interpolated_chw);

    cv::Mat interpolated_bgr, final_image;
    interpolated_chw.convertTo(interpolated_bgr, CV_8UC3, 255.0);

    cv::resize(interpolated_bgr, final_image, frame_before.size(), 0, 0, cv::INTER_CUBIC);

    return final_image;
}

} // namespace xinfer::zoo::generative