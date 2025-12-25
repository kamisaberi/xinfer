#pragma once
#include <xinfer/postproc/vision/types.h>
#include <vector>

namespace xinfer::postproc {

    struct TrackedObject {
        int track_id;       // Unique ID across time
        BoundingBox box;    // Current estimated position
        int age;            // How many frames alive
        int hits;           // How many detections matched
        float velocity_x;   // For leading shots
        float velocity_y;
    };

    struct TrackerConfig {
        int max_age = 30;       // Frames to keep track without detection
        int min_hits = 3;       // Frames to wait before confirming track
        float iou_threshold = 0.3f; // IoU for matching
    };

    class ITracker {
    public:
        virtual ~ITracker() = default;
        virtual void init(const TrackerConfig& config) = 0;

        /**
         * @brief Update tracks with new detections.
         * @param detections Current frame boxes (from YOLO).
         * @return List of active tracks.
         */
        virtual std::vector<TrackedObject> update(const std::vector<BoundingBox>& detections) = 0;

        /**
         * @brief Reset tracker state (e.g. new video file).
         */
        virtual void reset() = 0;
    };

}