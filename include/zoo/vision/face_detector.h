#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    /**
     * @brief Result structure for Face Detection.
     * Currently focuses on the Bounding Box.
     * (Future expansion: Landmarks/Keypoints support).
     */
    struct FaceResult {
        float x1, y1, x2, y2;
        float confidence;

        // Helper to get OpenCV Rect
        cv::Rect to_rect() const {
            return cv::Rect((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
        }
    };

    struct FaceDetectorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8n-face.rknn)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.45f;
        float nms_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FaceDetector {
    public:
        explicit FaceDetector(const FaceDetectorConfig& config);
        ~FaceDetector();

        // Move semantics
        FaceDetector(FaceDetector&&) noexcept;
        FaceDetector& operator=(FaceDetector&&) noexcept;
        FaceDetector(const FaceDetector&) = delete;
        FaceDetector& operator=(const FaceDetector&) = delete;

        /**
         * @brief Detect faces in an image.
         *
         * @param image Input image (BGR).
         * @return Vector of detected faces.
         */
        std::vector<FaceResult> detect(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision