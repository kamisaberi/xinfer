#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::hci {

    /**
     * @brief Result of Gaze Estimation.
     */
    struct GazeResult {
        // Gaze Vector in Camera Coordinate System
        // (x, y, z) unit vector pointing from eye center.
        cv::Point3f gaze_vector;

        // 2D Point of Regard (PoR)
        // Where the person is looking on the screen (pixel coordinates).
        cv::Point2f point_of_regard;

        float confidence;
    };

    struct GazeConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Face/Landmark Detector ---
        // (e.g. YOLO-Face with landmarks or a simple dlib model)
        std::string landmark_detector_path;

        // --- Model 2: Gaze Estimator ---
        // (e.g. GazeTR, L2CS-Net)
        std::string gaze_model_path;

        // Input Specs for Gaze Model
        int face_input_size = 224;
        int eye_input_size = 64;

        // Screen geometry (for Point of Regard calculation)
        // In meters, relative to camera.
        float screen_width_m = 0.5f;
        float screen_height_m = 0.3f;
        float screen_distance_m = 0.6f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class GazeTracker {
    public:
        explicit GazeTracker(const GazeConfig& config);
        ~GazeTracker();

        // Move semantics
        GazeTracker(GazeTracker&&) noexcept;
        GazeTracker& operator=(GazeTracker&&) noexcept;
        GazeTracker(const GazeTracker&) = delete;
        GazeTracker(const GazeTracker&) = delete;

        /**
         * @brief Estimate gaze from a single frame.
         *
         * Pipeline:
         * 1. Detect Face & Landmarks.
         * 2. Estimate Head Pose.
         * 3. Crop Face & Eyes.
         * 4. Inference on Gaze Model.
         * 5. Postprocess to get final vector.
         *
         * @param image Input video frame.
         * @return Gaze vector and point of regard.
         */
        GazeResult track(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::hci