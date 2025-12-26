#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::recycling {

    /**
     * @brief A detected waste item on the belt.
     */
    struct WasteItem {
        int track_id;           // Unique ID for tracking on conveyor
        std::string material;   // "PET_Clear", "HDPE", "Aluminum", "Cardboard"
        float confidence;

        postproc::BoundingBox box;
        cv::Point2f center;     // Calculated center for grasping/ejection

        bool in_ejection_zone;  // True if currently passing an actuator
        int target_bin_id;      // Which bin this should go to
    };

    struct SorterConfig {
        // Hardware Target (Low latency is critical for fast belts)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolo_waste.rknn)
        std::string model_path;

        // Label Map (Class ID -> Material Name)
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.5f;
        float nms_threshold = 0.45f;

        // Conveyor Logic
        // Define Y-coordinate lines where ejectors are placed
        // Map: Bin ID -> Y-Pixel Line
        std::map<int, int> ejection_lines;
        int ejection_tolerance_px = 50; // Zone height

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class Sorter {
    public:
        explicit Sorter(const SorterConfig& config);
        ~Sorter();

        // Move semantics
        Sorter(Sorter&&) noexcept;
        Sorter& operator=(Sorter&&) noexcept;
        Sorter(const Sorter&) = delete;
        Sorter& operator=(const Sorter&) = delete;

        /**
         * @brief Process a frame from the conveyor camera.
         *
         * Pipeline:
         * 1. Detect waste items.
         * 2. Track items (compensate for belt motion).
         * 3. Determine if item needs ejection.
         *
         * @param image Input frame.
         * @return List of active items on the belt.
         */
        std::vector<WasteItem> process(const cv::Mat& image);

        /**
         * @brief Reset tracking (e.g., belt stop/start).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::recycling