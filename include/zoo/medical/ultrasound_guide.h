#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::medical {

    /**
     * @brief Real-time guidance metrics.
     */
    struct GuideResult {
        // Visualization: Original image + Colored masks + Trajectory line
        cv::Mat overlay;

        // Distance from needle tip to target surface (mm)
        float distance_to_target_mm;

        // Safety Alert
        // True if needle trajectory intersects an artery/danger zone
        bool warning_collision;

        // Detected components
        bool target_visible;
        bool needle_visible;
    };

    struct UltrasoundConfig {
        // Hardware Target (Low latency is critical - <30ms preferred)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_us_nerve.engine)
        // Expected Output: Segmentation Mask (Class 0=Bg, 1=Target/Nerve, 2=Artery/Danger, 3=Needle)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Calibration
        float mm_per_pixel = 0.1f; // Depends on ultrasound depth setting

        // Class Mapping
        int class_id_target = 1;
        int class_id_danger = 2;
        int class_id_needle = 3;

        // Visualization
        float alpha_blend = 0.4f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class UltrasoundGuide {
    public:
        explicit UltrasoundGuide(const UltrasoundConfig& config);
        ~UltrasoundGuide();

        // Move semantics
        UltrasoundGuide(UltrasoundGuide&&) noexcept;
        UltrasoundGuide& operator=(UltrasoundGuide&&) noexcept;
        UltrasoundGuide(const UltrasoundGuide&) = delete;
        UltrasoundGuide& operator=(const UltrasoundGuide&) = delete;

        /**
         * @brief Process an ultrasound frame.
         *
         * Pipeline:
         * 1. Segmentation (Identify anatomy + needle).
         * 2. Vectorization (Fit line to needle mask).
         * 3. Trajectory Projection (Check collisions).
         *
         * @param image Input US frame (Grayscale).
         * @return Guidance overlay and metrics.
         */
        GuideResult process(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical