#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::geospatial {

    using RoadSegment = std::vector<cv::Point>;

    struct RoadExtractorConfig {
        std::string engine_path;
        int input_width = 1024;
        int input_height = 1024;
        float probability_threshold = 0.5f;
    };

    class RoadExtractor {
    public:
        explicit RoadExtractor(const RoadExtractorConfig& config);
        ~RoadExtractor();

        RoadExtractor(const RoadExtractor&) = delete;
        RoadExtractor& operator=(const RoadExtractor&) = delete;
        RoadExtractor(RoadExtractor&&) noexcept;
        RoadExtractor& operator=(RoadExtractor&&) noexcept;

        cv::Mat predict_mask(const cv::Mat& satellite_image);
        std::vector<RoadSegment> predict_segments(const cv::Mat& satellite_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial

