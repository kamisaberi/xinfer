#include <include/zoo/medical/cell_segmenter.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::medical {

struct CellSegmenter::Impl {
    CellSegmenterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

CellSegmenter::CellSegmenter(const CellSegmenterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Cell segmenter engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        // Normalization depends on the specific microscopy modality
        std::vector<float>{0.5f, 0.5f, 0.5f},
        std::vector<float>{0.5f, 0.5f, 0.5f}
    );
}

CellSegmenter::~CellSegmenter() = default;
CellSegmenter::CellSegmenter(CellSegmenter&&) noexcept = default;
CellSegmenter& CellSegmenter::operator=(CellSegmenter&&) noexcept = default;

CellSegmentationResult CellSegmenter::predict(const cv::Mat& microscope_image) {
    if (!pimpl_) throw std::runtime_error("CellSegmenter is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(microscope_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    auto output_shape = logit_map_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> logits(logit_map_tensor.num_elements());
    logit_map_tensor.copy_to_host(logits.data());

    cv::Mat probability_map(H, W, CV_32F, logits.data());

    cv::exp(-probability_map, probability_map);
    probability_map = 1.0 / (1.0 + probability_map);

    cv::Mat binary_mask;
    cv::threshold(probability_map, binary_mask, pimpl_->config_.probability_threshold, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8UC1);

    cv::Mat final_mask;
    cv::resize(binary_mask, final_mask, microscope_image.size(), 0, 0, cv::INTER_NEAREST);

    CellSegmentationResult result;
    cv::Mat labels;
    result.cell_count = cv::connectedComponents(final_mask, labels, 8, CV_32S);
    result.instance_mask = labels;

    cv::findContours(final_mask, result.contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    return result;
}

} // namespace xinfer::zoo::medical