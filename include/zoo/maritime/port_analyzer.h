#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::maritime {

    /**
     * @brief A tracked object within the port.
     */
    struct PortObject {
        int track_id;
        std::string category; // "Vessel", "Container", "Crane", "Vehicle"
        cv::Rect box;
        float confidence;
        int dwell_time_frames; // How long this object has been stationary
    };

    /**
     * @brief High-level KPIs (Key Performance Indicators) for the port.
     */
    struct PortAnalytics {
        // Counts
        int num_vessels_at_berth;
        int num_trucks_waiting;
        int num_containers_stacked;

        // Efficiency
        float average_truck_wait_time_sec;
        float quay_occupancy_percent;

        // Visualization
        cv::Mat overview_image;
    };

    struct PortConfig {
        // Hardware Target (Edge server with GPU is common for port-wide surveillance)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., yolov8_maritime.engine)
        // Trained on a dataset like SeaShips or a custom port dataset
        std::string model_path;

        // Label map for the detector
        std::string labels_path;

        // Input Specs
        int input_width = 1280;
        int input_height = 720;

        // Detection Settings
        float conf_threshold = 0.4f;
        float nms_threshold = 0.5f;

        // Analysis Zones (Defined as normalized rects [0-1])
        cv::Rect2f berth_zone;
        cv::Rect2f truck_queue_zone;
        cv::Rect2f container_yard_zone;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PortAnalyzer {
    public:
        explicit PortAnalyzer(const PortConfig& config);
        ~PortAnalyzer();

        // Move semantics
        PortAnalyzer(PortAnalyzer&&) noexcept;
        PortAnalyzer& operator=(PortAnalyzer&&) noexcept;
        PortAnalyzer(const PortAnalyzer&) = delete;
        PortAnalyzer& operator=(const PortAnalyzer&) = delete;

        /**
         * @brief Analyze a port surveillance frame.
         *
         * @param image Input frame.
         * @param fps Frame rate of the video (for time calculations).
         * @return High-level analytics and visualization.
         */
        PortAnalytics analyze(const cv::Mat& image, float fps);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::maritime