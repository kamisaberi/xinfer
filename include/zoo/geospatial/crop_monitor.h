#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::zoo::geospatial {

    struct CropMonitorConfig {
        // Config for a regression model that predicts health from spectral bands
        std::string engine_path;
        int input_width = 256;
        int input_height = 256;
    };

    class CropMonitor {
    public:
        explicit CropMonitor(const CropMonitorConfig& config);
        ~CropMonitor();

        CropMonitor(const CropMonitor&) = delete;
        CropMonitor& operator=(const CropMonitor&) = delete;
        CropMonitor(CropMonitor&&) noexcept;
        CropMonitor& operator=(CropMonitor&&) noexcept;

        cv::Mat predict_health_map(const cv::Mat& multispectral_image);

        static cv::Mat calculate_ndvi(const cv::Mat& multispectral_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial

