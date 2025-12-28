#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::aec {

    enum class SafetyViolation {
        NONE,
        MISSING_HARD_HAT,
        MISSING_VEST,
        PROXIMITY_BREACH,
        FALL_DETECTED
    };

    struct SafetyAlert {
        int worker_id; // Tracked ID
        SafetyViolation violation;
        postproc::BoundingBox box;
        float confidence;
    };

    struct SiteStatus {
        std::vector<SafetyAlert> alerts;
        bool is_safe;

        // Visualization
        cv::Mat annotated_image;
    };

    struct SafetyConfig {
        // Hardware Target (Edge NPU/GPU on-site)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model Paths ---
        // 1. Person/PPE Detector (e.g., yolov8_ppe.engine)
        std::string ppe_model_path;
        std::string ppe_labels_path; // "Person", "Hardhat", "Vest"

        // 2. Heavy Machinery Detector (YOLO)
        std::string machinery_model_path;
        std::string machinery_labels_path; // "Excavator", "Crane"

        // 3. Fall Detection (Pose Estimator) - Optional
        std::string pose_model_path;

        // --- Logic ---
        float ppe_conf_thresh = 0.5f;
        float machinery_conf_thresh = 0.6f;
        float safe_distance_m = 5.0f; // Minimum distance from machinery
        float pixels_per_meter = 20.0f; // Rough calibration

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SiteSafetyMonitor {
    public:
        explicit SiteSafetyMonitor(const SafetyConfig& config);
        ~SiteSafetyMonitor();

        // Move semantics
        SiteSafetyMonitor(SiteSafetyMonitor&&) noexcept;
        SiteSafetyMonitor& operator=(SiteSafetyMonitor&&) noexcept;
        SiteSafetyMonitor(const SiteSafetyMonitor&) = delete;
        SiteSafetyMonitor& operator=(const SiteSafetyMonitor&) = delete;

        /**
         * @brief Monitor a construction site frame for safety issues.
         *
         * @param image Input camera frame.
         * @return Site status report.
         */
        SiteStatus monitor(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::aec