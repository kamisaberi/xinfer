#pragma once


#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::vision {

    struct Keypoint {
        float x, y;
        float confidence;
    };

    using Pose = std::vector<Keypoint>;

    struct PoseEstimatorConfig {
        std::string engine_path;
        int input_width = 256;
        int input_height = 256;
        int num_keypoints = 17; // Default for COCO dataset
        float keypoint_threshold = 0.2f;
    };

    class PoseEstimator {
    public:
        explicit PoseEstimator(const PoseEstimatorConfig& config);
        ~PoseEstimator();

        PoseEstimator(const PoseEstimator&) = delete;
        PoseEstimator& operator=(const PoseEstimator&) = delete;
        PoseEstimator(PoseEstimator&&) noexcept;
        PoseEstimator& operator=(PoseEstimator&&) noexcept;

        std::vector<Pose> predict(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision

