#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

namespace xinfer::core { class Tensor; }
namespace xinfer::preproc { class ImageProcessor; }

namespace xinfer::zoo::medical {

    struct Tumor {
        int class_id;
        float confidence;
        std::string label;
        // 3D Bounding Box: center_x, center_y, center_z, width, height, depth
        float cx, cy, cz, w, h, d;
    };

    struct TumorDetectorConfig {
        std::string engine_path;
        std::string labels_path = "";
        float confidence_threshold = 0.5f;
        float nms_iou_threshold = 0.1f;
        int input_depth = 128;
        int input_height = 128;
        int input_width = 128;
    };

    class TumorDetector {
    public:
        explicit TumorDetector(const TumorDetectorConfig& config);
        ~TumorDetector();

        TumorDetector(const TumorDetector&) = delete;
        TumorDetector& operator=(const TumorDetector&) = delete;
        TumorDetector(TumorDetector&&) noexcept;
        TumorDetector& operator=(TumorDetector&&) noexcept;

        std::vector<Tumor> predict(const std::vector<cv::Mat>& ct_scan_slices);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical

