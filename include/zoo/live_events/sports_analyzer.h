#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::live_events {

    /**
     * @brief A single player on the field.
     */
    struct Player {
        int track_id;
        int team_id; // 0=Team A, 1=Team B, 2=Referee
        cv::Point2f position; // Center of box
        float speed_mps;
        std::vector<cv::Point2f> trajectory; // History of positions
        bool has_ball;
    };

    /**
     * @brief High-level tactical overview.
     */
    struct GameAnalytics {
        std::vector<Player> team_a;
        std::vector<Player> team_b;

        // Ball State
        bool is_ball_visible;
        cv::Point2f ball_position;
        int player_in_possession_id; // -1 if no one

        // Team Stats
        float team_a_possession_percent; // Running total

        // Visualization
        cv::Mat tactical_view;
    };

    struct SportsConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // --- Model 1: Detector (Player, Ball, Ref) ---
        std::string detector_path;
        int input_width = 1280;
        int input_height = 720;

        // --- Class Mappings ---
        int team_a_class_id = 0;
        int team_b_class_id = 1;
        int ball_class_id = 2;
        int ref_class_id = 3;

        // --- Physics Calibration ---
        // Projective Transform to map image pixels to field coordinates (meters)
        // Homography matrix from camera view to top-down "tactical" view
        cv::Mat homography_matrix;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SportsAnalyzer {
    public:
        explicit SportsAnalyzer(const SportsConfig& config);
        ~SportsAnalyzer();

        // Move semantics
        SportsAnalyzer(SportsAnalyzer&&) noexcept;
        SportsAnalyzer& operator=(SportsAnalyzer&&) noexcept;
        SportsAnalyzer(const SportsAnalyzer&) = delete;
        SportsAnalyzer& operator=(const SportsAnalyzer&) = delete;

        /**
         * @brief Analyze a frame of sports footage.
         *
         * @param image Input video frame.
         * @return Tactical game state.
         */
        GameAnalytics analyze(const cv::Mat& image);

        /**
         * @brief Reset all tracking and statistics.
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::live_events