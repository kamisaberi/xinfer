#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::medical {

    struct ArteryNode {
        cv::Point2f position;
        float diameter_mm; // Calculated width at this point
    };

    struct StenosisEvent {
        cv::Point2f location;
        float blockage_percentage; // 0% to 100%
        float healthy_diameter_mm;
        float actual_diameter_mm;
    };

    struct ArteryResult {
        // Binary mask of the vessel tree
        cv::Mat segmentation_mask;

        // Extracted centerlines with diameter info
        std::vector<std::vector<ArteryNode>> branches;

        // Detected abnormalities
        std::vector<StenosisEvent> stenoses;
    };

    struct ArteryConfig {
        // Hardware Target (Medical edge devices often use NVIDIA Clara or Intel OpenVINO)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_vessel_seg.engine)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Physics calibration
        float mm_per_pixel = 0.1f; // Calibration factor for real-world measurements

        // Sensitivity
        float seg_threshold = 0.5f;     // Binary threshold for segmentation
        float stenosis_threshold = 0.4f; // Flag if diameter drops by > 40%

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ArteryAnalyzer {
    public:
        explicit ArteryAnalyzer(const ArteryConfig& config);
        ~ArteryAnalyzer();

        // Move semantics
        ArteryAnalyzer(ArteryAnalyzer&&) noexcept;
        ArteryAnalyzer& operator=(ArteryAnalyzer&&) noexcept;
        ArteryAnalyzer(const ArteryAnalyzer&) = delete;
        ArteryAnalyzer& operator=(const ArteryAnalyzer&) = delete;

        /**
         * @brief Analyze a medical image frame.
         *
         * Pipeline:
         * 1. Segmentation (UNet).
         * 2. Skeletonization (Centerline extraction).
         * 3. Geometric Analysis (Distance Transform -> Diameter Profiling).
         *
         * @param image Input grayscale medical image (Angiogram).
         * @return Analysis metrics and overlays.
         */
        ArteryResult analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical