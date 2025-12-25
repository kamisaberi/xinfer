#pragma once

#include <xinfer/postproc/vision/tracker_interface.h>
#include <vector>
#include <memory>

namespace xinfer::postproc {

    /**
     * @brief CPU Object Tracker (SORT - Simple Online and Realtime Tracking).
     *
     * Logic:
     * 1. Kalman Filter predicts where objects should be in the next frame.
     * 2. Hungarian/Greedy matching associates new detections to predictions via IoU.
     * 3. Unmatched detections start new tracks.
     * 4. Unmatched tracks age and eventually die.
     */
    class CpuTracker : public ITracker {
    public:
        CpuTracker();
        ~CpuTracker() override;

        void init(const TrackerConfig& config) override;
        void reset() override;

        std::vector<TrackedObject> update(const std::vector<BoundingBox>& detections) override;

    private:
        struct Impl;
        std::unique_ptr<Impl> m_impl; // PImpl to hide OpenCV headers
    };

} // namespace xinfer::postproc