#include <include/zoo/vision/change_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::vision {

struct ChangeDetector::Impl {
    ChangeDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

ChangeDetector::ChangeDetector(const ChangeDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Change detection engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (pimpl_->engine_->get_num_inputs() != 2) {
        throw std::runtime_error("Change detection engine must have exactly two inputs.");
    }

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        pimpl_->config_.mean,
        pimpl_->config_.std
    );
}

ChangeDetector::~ChangeDetector() = default;
ChangeDetector::ChangeDetector(ChangeDetector&&) noexcept = default;
ChangeDetector& ChangeDetector::operator=(ChangeDetector&&) noexcept = default;

cv::Mat ChangeDetector::predict(const cv::Mat& image_before, const cv::Mat& image_after) {
    if (!pimpl_) throw std::runtime_error("ChangeDetector is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor_before(input_shape, core::DataType::kFLOAT);
    core::Tensor input_tensor_after(input_shape, core::DataType::kFLOAT);

    pimpl_->preprocessor_->process(image_before, input_tensor_before);
    pimpl_->preprocessor_->process(image_after, input_tensor_after);

    auto output_tensors = pimpl_->engine_->infer({input_tensor_before, input_tensor_after});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    auto output_shape = logit_map_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> logits(logit_map_tensor.num_elements());
    logit_map_tensor.copy_to_host(logits.data());

    cv::Mat change_mask(H, W, CV_8UC1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float logit_val = logits[y * W + x];
            float probability = 1.0f / (1.0f + expf(-logit_val));
            change_mask.at<uchar>(y, x) = (probability > pimpl_->config_.change_threshold) ? 255 : 0;
        }
    }

    cv::Mat final_mask;
    cv::resize(change_mask, final_mask, image_before.size(), 0, 0, cv::INTER_NEAREST);

    return final_mask;
}

} // namespace xinfer::zoo::vision