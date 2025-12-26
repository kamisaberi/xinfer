#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::geospatial {

    struct Vessel {
        std::string type;     // "Cargo Ship", "Sailboat", "Buoy"
        float confidence;
        postproc::BoundingBox box;

        // Optional tracking ID
        int track_id = -1;
    };

    struct MaritimeConfig {
        // Hardware Target (Edge GPU/NPU for onboard systems)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., yolov8_seaships.engine)
        std::string model_path;

        // Label map for the detector
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.45f;
        float nms_threshold = 0.5f;

        // Enable tracking?
        bool enable_tracking = true;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class MaritimeDetector {
    public:
        explicit MaritimeDetector(const MaritimeConfig& config);
        ~MaritimeDetector();

        // Move semantics
        MaritimeDetector(MaritimeDetector&&) noexcept;
        MaritimeDetector& operator=(MaritimeDetector&&) noexcept;
        MaritimeDetector(const MaritimeDetector&) = delete;
        MaritimeDetector& operator=(const MaritimeDetector&) = delete;

        /**
         * @brief Detect and/or track vessels in a frame.
         *
         * @param image Input frame.
         * @return List of detected vessels.
         */
        std::vector<Vessel> detect(const cv::Mat& image);

        /**
         * @brief Reset tracking state.
         */
        void reset_tracker();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::geospatial