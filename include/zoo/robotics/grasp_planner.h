#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::robotics {

    /**
     * @brief Represents a 6-DoF Grasp candidate.
     */
    struct Grasp {
        // Position in Camera Frame (meters)
        float x, y, z;

        // Orientation (Radians around Z-axis)
        float angle;

        // Gripper opening width (meters)
        float width;

        // Quality Score (0.0 - 1.0)
        float score;

        // Pixel coordinates (for visualization)
        int u, v;
    };

    struct GraspConfig {
        // Hardware Target (Jetson or Intel CPU/iGPU often used for arms)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., ggcnn.onnx, gr_convnet.engine)
        std::string model_path;

        // Input Specs
        int input_width = 300;
        int input_height = 300;

        // Camera Intrinsics (Required for 2D->3D deprojection)
        float fx = 615.0f;
        float fy = 615.0f;
        float cx = 320.0f;
        float cy = 240.0f;

        // Depth Processing
        float depth_scale = 0.001f; // Convert raw integer depth to meters (e.g. mm -> m)
        float crop_size = 300.0f;   // Size of the central crop in pixels

        // Filtering
        float score_threshold = 0.5f;
        int max_grasps = 10;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class GraspPlanner {
    public:
        explicit GraspPlanner(const GraspConfig& config);
        ~GraspPlanner();

        // Move semantics
        GraspPlanner(GraspPlanner&&) noexcept;
        GraspPlanner& operator=(GraspPlanner&&) noexcept;
        GraspPlanner(const GraspPlanner&) = delete;
        GraspPlanner& operator=(const GraspPlanner&) = delete;

        /**
         * @brief Plan grasps from a depth image.
         *
         * Pipeline:
         * 1. Preprocess: Center crop, resize, normalize depth.
         * 2. Inference: Generates dense grasp maps (Q, Angle, Width).
         * 3. Postprocess: Find local maxima, decode angles, deproject to 3D.
         *
         * @param depth_image Input Depth image (CV_16U or CV_32F).
         * @return List of best grasp poses sorted by score.
         */
        std::vector<Grasp> plan(const cv::Mat& depth_image);

        /**
         * @brief Utility: Draw grasps on an RGB image for debug.
         */
        static void draw_grasps(cv::Mat& image, const std::vector<Grasp>& grasps);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::robotics