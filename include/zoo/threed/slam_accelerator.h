#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::threed {

    /**
     * @brief A set of features for a single frame.
     */
    struct FrameFeatures {
        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors; // [NumPoints, DescDim] (Float32)
        cv::Mat scores;      // Detection confidence per point
    };

    struct SlamConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Feature Extractor Model (e.g., superpoint.engine)
        std::string model_path;

        // Input Specs (Grayscale usually)
        int input_width = 640;
        int input_height = 480;

        // Keypoint Extraction Settings
        int max_keypoints = 1000;
        float keypoint_threshold = 0.015f;
        int nms_radius = 4; // Non-Maximum Suppression radius

        // Descriptor Dimension (SuperPoint=256)
        int descriptor_dim = 256;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SlamAccelerator {
    public:
        explicit SlamAccelerator(const SlamConfig& config);
        ~SlamAccelerator();

        // Move semantics
        SlamAccelerator(SlamAccelerator&&) noexcept;
        SlamAccelerator& operator=(SlamAccelerator&&) noexcept;
        SlamAccelerator(const SlamAccelerator&) = delete;
        SlamAccelerator& operator=(const SlamAccelerator&) = delete;

        /**
         * @brief Extract features from an image.
         *
         * Pipeline:
         * 1. Preprocess (Resize/Grayscale).
         * 2. Inference (Dense Feature Map).
         * 3. Postprocess (NMS + Bi-cubic Descriptor Sampling).
         *
         * @param image Input image.
         * @return Keypoints and Descriptors.
         */
        FrameFeatures extract(const cv::Mat& image);

        /**
         * @brief Utility: Match features between two frames.
         * Uses internal logic or OpenCV KNN to find correspondences.
         */
        static std::vector<cv::DMatch> match(const FrameFeatures& f1, const FrameFeatures& f2);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed