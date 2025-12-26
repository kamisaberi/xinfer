#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::geospatial {

    struct CropHealthStats {
        float percent_healthy;
        float percent_stressed;
        float percent_weeds;
        float percent_soil;

        // Key metric for yield prediction
        float ndvi_mean; // Normalized Difference Vegetation Index
    };

    struct CropResult {
        CropHealthStats stats;

        // Color-coded map
        // Green=Healthy, Yellow=Stressed, Red=Weeds, Brown=Soil
        cv::Mat health_map;
    };

    struct CropConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., unet_crop_health.onnx)
        // A model trained on multispectral data (e.g., RGB + NIR)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;
        int input_channels = 4; // RGB + NIR

        // Class Mapping
        // Must match model training order
        std::vector<std::string> class_names = {"Soil", "Healthy", "Stressed", "Weed"};

        // Channel indices for NDVI calculation (NIR, Red)
        int nir_channel_idx = 3;
        int red_channel_idx = 0;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CropMonitor {
    public:
        explicit CropMonitor(const CropConfig& config);
        ~CropMonitor();

        // Move semantics
        CropMonitor(CropMonitor&&) noexcept;
        CropMonitor& operator=(CropMonitor&&) noexcept;
        CropMonitor(const CropMonitor&) = delete;
        CropMonitor& operator=(const CropMonitor&) = delete;

        /**
         * @brief Analyze a multispectral image tile.
         *
         * @param image Input image (e.g., 4-channel CV_8UC4 or CV_32FC4).
         * @return Health statistics and visualization.
         */
        CropResult analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial