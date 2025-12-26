#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::space {

    /**
     * @brief 6-DoF Pose relative to the target port.
     * Coordinate Frame: Camera Frame (Z-forward, X-right, Y-down usually).
     */
    struct Pose6D {
        // Translation (meters)
        float tx, ty, tz;

        // Rotation (Euler angles in degrees, or Quaternion)
        float roll, pitch, yaw;

        // Alignment Metric (1.0 = Perfectly aligned for docking)
        float alignment_score;
    };

    enum class DockingStatus {
        SEARCHING = 0,      // Target not visible
        APPROACHING = 1,    // Target visible, far away
        TERMINAL = 2,       // Close range, precision mode
        LOCKED = 3          // Ready to latch
    };

    struct DockingResult {
        Pose6D pose;
        DockingStatus status;
        std::vector<cv::Point2f> keypoints_2d; // Detected features for viz
        float inference_time_ms;
    };

    struct DockingConfig {
        // Hardware Target (Radiation-hardened FPGA or Edge GPU)
        xinfer::Target target = xinfer::Target::AMD_VITIS;

        // Model Path (e.g., port_keypoints.xmodel)
        // Model should output 2D coordinates of N keypoints.
        std::string model_path;

        // Camera Intrinsics (Critical for PnP)
        float fx = 1000.0f;
        float fy = 1000.0f;
        float cx = 512.0f;
        float cy = 512.0f;

        // Target Geometry (Real-world 3D coordinates of the docking markers)
        // e.g., 4 corners of a 10cm square.
        std::vector<cv::Point3f> target_geometry;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DockingController {
    public:
        explicit DockingController(const DockingConfig& config);
        ~DockingController();

        // Move semantics
        DockingController(DockingController&&) noexcept;
        DockingController& operator=(DockingController&&) noexcept;
        DockingController(const DockingController&) = delete;
        DockingController& operator=(const DockingController&) = delete;

        /**
         * @brief Calculate relative pose to the docking target.
         *
         * Pipeline:
         * 1. Inference: Detect 2D keypoints of the target.
         * 2. Math: Solve PnP (2D detections -> 3D Geometry).
         * 3. Logic: Determine Approach Status.
         *
         * @param image Camera frame.
         * @return 6-DoF Pose and Status.
         */
        DockingResult calculate_pose(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::space