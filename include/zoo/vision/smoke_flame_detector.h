#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    enum class HazardType {
        FIRE = 0,
        SMOKE = 1,
        UNKNOWN = 2
    };

    struct HazardResult {
        HazardType type;
        float confidence;
        postproc::BoundingBox box;
    };

    struct SmokeFlameConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8n_fire_smoke.rknn)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Sensitivity Settings
        // Fire detection needs high recall, so defaults are often lower than generic detection.
        float fire_thresh = 0.35f;
        float smoke_thresh = 0.35f;
        float nms_threshold = 0.45f;

        // Class Mapping (Model specific)
        // Which class ID corresponds to Fire/Smoke in the model?
        int class_id_fire = 0;
        int class_id_smoke = 1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SmokeFlameDetector {
    public:
        explicit SmokeFlameDetector(const SmokeFlameConfig& config);
        ~SmokeFlameDetector();

        // Move semantics
        SmokeFlameDetector(SmokeFlameDetector&&) noexcept;
        SmokeFlameDetector& operator=(SmokeFlameDetector&&) noexcept;
        SmokeFlameDetector(const SmokeFlameDetector&) = delete;
        SmokeFlameDetector& operator=(const SmokeFlameDetector&) = delete;

        /**
         * @brief Detect fire and smoke in an image.
         *
         * @param image Input frame.
         * @return List of detected hazards.
         */
        std::vector<HazardResult> detect(const cv::Mat& image);

        /**
         * @brief Utility: Draw alerts on image.
         * Draws Red boxes for Fire, Grey boxes for Smoke.
         */
        static void draw_alerts(cv::Mat& image, const std::vector<HazardResult>& results);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision