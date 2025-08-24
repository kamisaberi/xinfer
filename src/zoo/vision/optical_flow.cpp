#include <include/zoo/vision/optical_flow.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct OpticalFlow::Impl {
    OpticalFlowConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

OpticalFlow::OpticalFlow(const OpticalFlowConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Optical flow engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Optical flow engine must have exactly two inputs.");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // RAFT models typically use [0, 255] range, no mean/std normalization
        std::vector<float>{0.0f, 0.0f, 0.0f},
        std::vector<float>{1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f}
    );
}

OpticalFlow::~OpticalFlow() = default;
OpticalFlow::OpticalFlow(OpticalFlow&&) noexcept = default;
OpticalFlow& OpticalFlow::operator=(OpticalFlow&&) noexcept = default;

cv::Mat OpticalFlow::predict(const cv::Mat& frame1, const cv::Mat& frame2) {
    if (!pimpl_) throw std::runtime_error("OpticalFlow is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor1(input_shape, core::DataType::kFLOAT);
    core::Tensor input_tensor2(input_shape, core::DataType::kFLOAT);

    pimpl_->preprocessor_->process(frame1, input_tensor1);
    pimpl_->preprocessor_->process(frame2, input_tensor2);

    auto output_tensors = pimpl_->engine_->infer({input_tensor1, input_tensor2});
    const core::Tensor& flow_map_tensor = output_tensors[0];

    auto output_shape = flow_map_tensor.shape();
    const int C = output_shape[1];
    const int H = output_shape[2];
    const int W = output_shape[3];

    if (C != 2) {
        throw std::runtime_error("Optical flow model output must have 2 channels (flow_x, flow_y).");
    }

    std::vector<float> h_output(flow_map_tensor.num_elements());
    flow_map_tensor.copy_to_host(h_output.data());

    cv::Mat flow_map(H, W, CV_32FC2);

    std::vector<cv::Mat> channels;
    channels.emplace_back(H, W, CV_32F, h_output.data());
    channels.emplace_back(H, W, CV_32F, h_output.data() + H * W);
    cv::merge(channels, flow_map);

    cv::Mat final_flow_map;
    cv::resize(flow_map, final_flow_map, frame1.size(), 0, 0, cv::INTER_LINEAR);

    final_flow_map.at<cv::Vec2f>(0,0)[0] *= (float)frame1.cols / (float)W;
    final_flow_map.at<cv::Vec2f>(0,0)[1] *= (float)frame1.rows / (float)H;

    return final_flow_map;
}

} // namespace xinfer::zoo::vision