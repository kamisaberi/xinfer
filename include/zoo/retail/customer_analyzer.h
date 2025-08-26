#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/zoo/vision/pose_estimator.h>

namespace xinfer::zoo::retail {

    struct TrackedCustomer {
        int track_id;
        cv::Rect bounding_box;
        vision::Pose pose;
    };

    struct CustomerAnalyzerConfig {
        std::string detection_engine_path; // For person detection
        std::string pose_engine_path;      // For pose estimation
        float detection_confidence_threshold = 0.6f;
        float detection_nms_iou_threshold = 0.5f;
        int detection_input_width = 640;
        int detection_input_height = 480;
    };

    class CustomerAnalyzer {
    public:
        explicit CustomerAnalyzer(const CustomerAnalyzerConfig& config);
        ~CustomerAnalyzer();

        CustomerAnalyzer(const CustomerAnalyzer&) = delete;
        CustomerAnalyzer& operator=(const CustomerAnalyzer&) = delete;
        CustomerAnalyzer(CustomerAnalyzer&&) noexcept;
        CustomerAnalyzer& operator=(CustomerAnalyzer&&) noexcept;

        std::vector<TrackedCustomer> track(const cv::Mat& frame);
        cv::Mat generate_heatmap();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail

