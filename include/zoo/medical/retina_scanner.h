#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct RetinaScanResult {
        int severity_grade; // 0 (None) to 4 (Proliferative DR)
        float confidence;
        std::string diagnosis;
        cv::Mat lesion_heatmap;
    };

    struct RetinaScannerConfig {
        std::string engine_path;
        int input_width = 512;
        int input_height = 512;
    };

    class RetinaScanner {
    public:
        explicit RetinaScanner(const RetinaScannerConfig& config);
        ~RetinaScanner();

        RetinaScanner(const RetinaScanner&) = delete;
        RetinaScanner& operator=(const RetinaScanner&) = delete;
        RetinaScanner(RetinaScanner&&) noexcept;
        RetinaScanner& operator=(RetinaScanner&&) noexcept;

        RetinaScanResult predict(const cv::Mat& fundus_image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical

