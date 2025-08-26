#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct ArteryAnalysisResult {
        cv::Mat vessel_mask;
        float stenosis_score; // A score from 0.0 (healthy) to 1.0 (severe blockage)
        std::vector<cv::Point> stenosis_location;
    };

    struct ArteryAnalyzerConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
        float probability_threshold = 0.5f;
    };

    class ArteryAnalyzer {
    public:
        explicit ArteryAnalyzer(const ArteryAnalyzerConfig& config);
        ~ArteryAnalyzer();

        ArteryAnalyzer(const ArteryAnalyzer&) = delete;
        ArteryAnalyzer& operator=(const ArteryAnalyzer&) = delete;
        ArteryAnalyzer(ArteryAnalyzer&&) noexcept;
        ArteryAnalyzer& operator=(ArteryAnalyzer&&) noexcept;

        ArteryAnalysisResult predict(const cv::Mat& angiogram_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical

