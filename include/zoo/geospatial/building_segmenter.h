#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::geospatial {

    using BuildingPolygon = std::vector<cv::Point>;

    struct BuildingSegmenterConfig {
        std::string engine_path;
        int input_width = 1024;
        int input_height = 1024;
        float probability_threshold = 0.5f;
    };

    class BuildingSegmenter {
    public:
        explicit BuildingSegmenter(const BuildingSegmenterConfig& config);
        ~BuildingSegmenter();

        BuildingSegmenter(const BuildingSegmenter&) = delete;
        BuildingSegmenter& operator=(const BuildingSegmenter&) = delete;
        BuildingSegmenter(BuildingSegmenter&&) noexcept;
        BuildingSegmenter& operator=(BuildingSegmenter&&) noexcept;

        cv::Mat predict_mask(const cv::Mat& satellite_image);
        std::vector<BuildingPolygon> predict_polygons(const cv::Mat& satellite_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial

