#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct CellSegmentationResult {
        int cell_count;
        cv::Mat instance_mask; // Each cell instance has a unique integer ID
        std::vector<std::vector<cv::Point>> contours;
    };

    struct CellSegmenterConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
        float probability_threshold = 0.5f;
    };

    class CellSegmenter {
    public:
        explicit CellSegmenter(const CellSegmenterConfig& config);
        ~CellSegmenter();

        CellSegmenter(const CellSegmenter&) = delete;
        CellSegmenter& operator=(const CellSegmenter&) = delete;
        CellSegmenter(CellSegmenter&&) noexcept;
        CellSegmenter& operator=(CellSegmenter&&) noexcept;

        CellSegmentationResult predict(const cv::Mat& microscope_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical

