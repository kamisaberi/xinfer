#pragma once

#include <string>
#include <vector>
#include <memory>
#include <set>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Represents a single tracked animal in the current frame.
     */
    struct AnimalTrack {
        int track_id;       // Unique persistent ID (e.g., 42)
        float x1, y1, x2, y2; // Bounding box
        float confidence;
        std::string species; // Label (e.g., "Bear", "Deer")

        // Motion estimation (useful for predicting movement)
        float velocity_x;
        float velocity_y;
    };

    struct AnimalTrackerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (YOLO detection model trained on animals)
        std::string model_path;

        // Path to class labels file (coco.names or custom animal list)
        std::string labels_path;

        // Inference Settings
        int input_width = 640;
        int input_height = 640;
        float conf_threshold = 0.45f;
        float nms_threshold = 0.5f;

        // Tracking Settings
        int max_track_age = 30; // Frames to keep ID alive if occluded
        int min_hits = 3;       // Frames before confirming ID

        // Optional: Filter specific class IDs from the model
        // e.g. {15, 16, 17} for COCO (Cat, Dog, Horse)
        // If empty, tracks all detected objects.
        std::set<int> filter_class_ids;

        // Vendor flags (e.g. "CORE=0")
        std::vector<std::string> vendor_params;
    };

    class AnimalTracker {
    public:
        explicit AnimalTracker(const AnimalTrackerConfig& config);
        ~AnimalTracker();

        // Move semantics
        AnimalTracker(AnimalTracker&&) noexcept;
        AnimalTracker& operator=(AnimalTracker&&) noexcept;
        AnimalTracker(const AnimalTracker&) = delete;
        AnimalTracker& operator=(const AnimalTracker&) = delete;

        /**
         * @brief Process a video frame.
         *
         * Pipeline:
         * 1. Preprocess (Resize/Norm)
         * 2. Detect (YOLO inference)
         * 3. Decode & NMS
         * 4. Update Tracker (Kalman Filter)
         *
         * @return List of active animal tracks.
         */
        std::vector<AnimalTrack> track(const cv::Mat& image);

        /**
         * @brief Reset tracker state (e.g., when switching video files).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision