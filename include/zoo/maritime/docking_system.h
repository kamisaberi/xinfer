#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::maritime {

    /**
     * @brief Pose of the vessel relative to the target berth.
     */
    struct DockingState {
        // Distance from bow to pier line (meters)
        float distance_to_pier;

        // Lateral offset from centerline of berth (meters)
        float lateral_offset;

        // Relative angle to pier (degrees)
        // 0 = Parallel, +ve = Bow-in
        float angle_to_pier;

        bool is_aligned; // True if all metrics are within tolerance
    };

    struct DockingResult {
        DockingState state;

        // Visualization overlay with detected lines
        cv::Mat visualization;
    };

    struct DockingSystemConfig {
        // Hardware Target (NVIDIA Jetson / Intel iGPU common)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., unet_water_segmentation.onnx)
        // Model separates Water vs. Pier/Land
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Camera Calibration (For Pixel -> Meter conversion)
        float mm_per_pixel = 100.0f; // Simplified for this example
        float focal_length_px = 800.0f;

        // Alignment Tolerances
        float distance_tolerance_m = 1.0f;
        float angle_tolerance_deg = 2.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DockingSystem {
    public:
        explicit DockingSystem(const DockingSystemConfig& config);
        ~DockingSystem();

        // Move semantics
        DockingSystem(DockingSystem&&) noexcept;
        DockingSystem& operator=(DockingSystem&&) noexcept;
        DockingSystem(const DockingSystem&) = delete;
        DockingSystem& operator=(const DockingSystem&) = delete;

        /**
         * @brief Analyze a camera frame to guide docking.
         *
         * Pipeline:
         * 1. Segment Image (Water vs Pier).
         * 2. Edge Detection (Canny).
         * 3. Line Fitting (Hough Transform).
         * 4. Geometric Calculation (Distance & Angle).
         *
         * @param image Camera frame.
         * @return Docking state and visualization.
         */
        DockingResult analyze_frame(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::maritime