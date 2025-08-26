#include <include/zoo/geospatial/building_segmenter.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <cmath>

#include <include/core/engine.h>
#include <include/preproc/image_processor.h>

namespace xinfer::zoo::geospatial {

struct BuildingSegmenter::Impl {
    BuildingSegmenterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::unique_ptr<preproc::ImageProcessor> preprocessor_;
};

BuildingSegmenter::BuildingSegmenter(const BuildingSegmenterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("Building segmenter engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    pimpl_->preprocessor_ = std::make_unique<preproc::ImageProcessor>(
        pimpl_->config_.input_width,
        pimpl_->config_.input_height,
        std::vector<float>{0.485f, 0.456f, 0.406f},
        std::vector<float>{0.229f, 0.224f, 0.225f}
    );
}

BuildingSegmenter::~BuildingSegmenter() = default;
BuildingSegmenter::BuildingSegmenter(BuildingSegmenter&&) noexcept = default;
BuildingSegmenter& BuildingSegmenter::operator=(BuildingSegmenter&&) noexcept = default;

cv::Mat BuildingSegmenter::predict_mask(const cv::Mat& satellite_image) {
    if (!pimpl_) throw std::runtime_error("BuildingSegmenter is in a moved-from state.");

    auto input_shape = pimpl_->engine_->get_input_shape(0);
    core::Tensor input_tensor(input_shape, core::DataType::kFLOAT);
    pimpl_->preprocessor_->process(satellite_image, input_tensor);

    auto output_tensors = pimpl_->engine_->infer({input_tensor});
    const core::Tensor& logit_map_tensor = output_tensors[0];

    auto output_shape = logit_map_tensor.shape();
    const int H = output_shape[2];
    const int W = output_shape[3];

    std::vector<float> logits(logit_map_tensor.num_elements());
    logit_map_tensor.copy_to_host(logits.data());

    cv::Mat probability_map(H, W, CV_32F, logits.data());

    // Assuming the output is a single-channel logit map for "building" class.
    // Apply sigmoid to get probabilities.
    cv::exp(-probability_map, probability_map);
    probability_map = 1.0 / (1.0 + probability_map);

    cv::Mat binary_mask;
    cv::threshold(probability_map, binary_mask, pimpl_->config_.probability_threshold, 255, cv::THRESH_BINARY);
    binary_mask.convertTo(binary_mask, CV_8UC1);

    cv::Mat final_mask;
    cv::resize(binary_mask, final_mask, satellite_image.size(), 0, 0, cv::INTER_NEAREST);

    return final_mask;
}

std::vector<BuildingPolygon> BuildingSegmenter::predict_polygons(const cv::Mat& satellite_image) {
    cv::Mat mask = predict_mask(satellite_image);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<BuildingPolygon> polygons;
    for(const auto& contour : contours) {
        BuildingPolygon polygon;
        cv::approxPolyDP(contour, polygon, 3, true);
        polygons.push_back(polygon);
    }

    return polygons;
}

} // namespace xinfer::zoo::geospatial