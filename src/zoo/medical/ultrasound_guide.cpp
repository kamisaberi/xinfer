#include <include/zoo/medical/ultrasound_guide.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/segmentation.h>

namespace xinfer::zoo::medical {

struct UltrasoundGuide::Impl {
    UltrasoundGuideConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

UltrasoundGuide::UltrasoundGuide(const UltrasoundGuideConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Ultrasound guide engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Ultrasound images are typically grayscale and normalized simply
        std::vector<float>{0.5f},
        std::vector<float>{0.5f}
    );
}

UltrasoundGuide::~UltrasoundGuide() = default;
UltrasoundGuide::UltrasoundGuide(UltrasoundGuide&&) noexcept = default;
UltrasoundGuide& UltrasoundGuide::operator=(UltrasoundGuide&&) noexcept = default;

UltrasoundGuideResult UltrasoundGuide::predict(const cv::Mat& ultrasound_image) {
    if (!pimpl_) throw std::runtime_error("UltrasoundGuide is in a moved-from state.");

    cv::Mat gray_image;
    if (ultrasound_image.channels() == 3) {
        cv::cvtColor(ultrasound_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = ultrasound_image;
    }

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(gray_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    cv::Mat class_mask = postproc::argmax_to_mat(logit_map_tensor);

    cv::Mat final_mask;
    cv::resize(class_mask, final_mask, ultrasound_image.size(), 0, 0, cv::INTER_NEAREST);
    final_mask.convertTo(final_mask, CV_8UC1, 255);

    UltrasoundGuideResult result;
    result.segmentation_mask = final_mask;

    cv::findContours(final_mask, result.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return result;
}

} // namespace xinfer::zoo::medical