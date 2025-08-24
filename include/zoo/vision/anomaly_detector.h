#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct AnomalyResult {
        bool is_anomaly;
        float anomaly_score;
        cv::Mat anomaly_map;
    };

    struct AnomalyDetectorConfig {
        std::string engine_path;
        float anomaly_threshold = 0.1f;
        int input_width = 256;
        int input_height = 256;
        std::vector<float> mean = {0.5f, 0.5f, 0.5f};
        std::vector<float> std = {0.5f, 0.5f, 0.5f};
    };

    class AnomalyDetector {
    public:
        explicit AnomalyDetector(const AnomalyDetectorConfig& config);
        ~AnomalyDetector();

        AnomalyDetector(const AnomalyDetector&) = delete;
        AnomalyDetector& operator=(const AnomalyDetector&) = delete;
        AnomalyDetector(AnomalyDetector&&) noexcept;
        AnomalyDetector& operator=(AnomalyDetector&&) noexcept;

        AnomalyResult predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

