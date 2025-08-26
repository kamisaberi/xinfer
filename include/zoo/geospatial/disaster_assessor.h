#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::geospatial {

    struct DisasterAssessorConfig {
        std::string engine_path;
        float damage_threshold = 0.5f;
        int input_width = 1024;
        int input_height = 1024;
        std::vector<float> mean = {0.485, 0.456, 0.406};
        std::vector<float> std = {0.229, 0.224, 0.225};
    };

    class DisasterAssessor {
    public:
        explicit DisasterAssessor(const DisasterAssessorConfig& config);
        ~DisasterAssessor();

        DisasterAssessor(const DisasterAssessor&) = delete;
        DisasterAssessor& operator=(const DisasterAssessor&) = delete;
        DisasterAssessor(DisasterAssessor&&) noexcept;
        DisasterAssessor& operator=(DisasterAssessor&&) noexcept;

        cv::Mat predict(const cv::Mat& image_before, const cv::Mat& image_after);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial

