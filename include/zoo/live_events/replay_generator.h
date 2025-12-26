#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <deque>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::live_events {

    /**
     * @brief A detected highlight event.
     */
    struct HighlightEvent {
        std::string event_type; // "Goal", "Foul", "Corner_Kick"
        float confidence;
        long long timestamp_ms; // When the event occurred

        // The frames that make up the replay clip
        std::vector<cv::Mat> replay_frames;
    };

    struct ReplayConfig {
        // Hardware Target (GPU/NPU required for 3D CNNs)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., resnet3d_sports.engine)
        // A video action classifier
        std::string model_path;
        std::string labels_path; // Action labels ("Goal", "Pass", "Background")

        // Temporal Window
        int window_size = 16;      // Model takes 16 frames as input
        int frame_stride = 2;      // Sample every 2nd frame from live feed

        // Input Specs
        int input_width = 224;
        int input_height = 224;

        // Trigger Settings
        float event_threshold = 0.8f;

        // How many seconds of footage before the event to save?
        float pre_roll_sec = 5.0f;
        // How many seconds after?
        float post_roll_sec = 2.0f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ReplayGenerator {
    public:
        explicit ReplayGenerator(const ReplayConfig& config);
        ~ReplayGenerator();

        // Move semantics
        ReplayGenerator(ReplayGenerator&&) noexcept;
        ReplayGenerator& operator=(ReplayGenerator&&) noexcept;
        ReplayGenerator(const ReplayGenerator&) = delete;
        ReplayGenerator& operator=(const ReplayGenerator&) = delete;

        /**
         * @brief Process a live frame and check for highlight events.
         *
         * This function must be called for every frame of the stream.
         *
         * @param image The current video frame.
         * @param timestamp_ms Current stream time.
         * @return A HighlightEvent if a new one was just triggered, otherwise returns an empty/invalid event.
         */
        std::vector<HighlightEvent> process_frame(const cv::Mat& image, long long timestamp_ms);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::live_events

