#include <include/zoo/medical/artery_analyzer.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>
#include <include/postproc/segmentation.h>

namespace xinfer::zoo::medical {

struct ArteryAnalyzer::Impl {
    ArteryAnalyzerConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

ArteryAnalyzer::ArteryAnalyzer(const ArteryAnalyzerConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Artery analyzer engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Medical images often use simple scaling to [0,1]
        std::vector<float>{0.0f},
        std::vector<float>{1.0f}
    );
}

ArteryAnalyzer::~ArteryAnalyzer() = default;
ArteryAnalyzer::ArteryAnalyzer(ArteryAnalyzer&&) noexcept = default;
ArteryAnalyzer& ArteryAnalyzer::operator=(ArteryAnalyzer&&) noexcept = default;

ArteryAnalysisResult ArteryAnalyzer::predict(const cv::Mat& angiogram_image) {
    if (!pimpl_) throw std::runtime_error("ArteryAnalyzer is in a moved-from state.");

    cv::Mat gray_image;
    if (angiogram_image.channels() == 3) {
        cv::cvtColor(angiogram_image, gray_image, cv::COLOR_BGR2GRAY);
    } else {
        gray_image = angiogram_image;
    }

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(gray_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    cv::Mat class_mask = postproc::argmax_to_mat(logit_map_tensor);

    cv::Mat final_mask;
    cv::resize(class_mask, final_mask, angiogram_image.size(), 0, 0, cv::INTER_NEAREST);
    final_mask.convertTo(final_mask, CV_8UC1, 255);

    ArteryAnalysisResult result;
    result.vessel_mask = final_mask;
    result.stenosis_score = 0.0f; // Placeholder for analysis logic

    // A full implementation would have complex morphological analysis here
    // to find the narrowest points in the vessel mask to calculate a stenosis score.

    return result;
}

} // namespace xinfer::zoo::medical