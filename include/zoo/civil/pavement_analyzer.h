#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::civil {

    /**
     * @brief Statistics of the analyzed pavement section.
     */
    struct PavementStats {
        // Percentage of the road surface covered by each defect type
        float percent_cracks;
        float percent_potholes;
        float percent_rutting;

        // Overall Pavement Condition Index (PCI) proxy score
        // 1.0 = Perfect, 0.0 = Needs immediate repair
        float condition_score;
    };

    struct PavementResult {
        PavementStats stats;

        // Visualization
        cv::Mat defect_map; // Color-coded overlay
    };

    struct PavementConfig {
        // Hardware Target (Often run in batch on a server/GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., unet_pavement_defects.engine)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Class Mapping
        // Must match model's training order
        std::vector<std::string> class_names = {"Background", "Pavement", "Crack", "Pothole", "Rutting"};

        // Visualization Colors (BGR)
        std::vector<std::vector<uint8_t>> class_colors;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PavementAnalyzer {
    public:
        explicit PavementAnalyzer(const PavementConfig& config);
        ~PavementAnalyzer();

        // Move semantics
        PavementAnalyzer(PavementAnalyzer&&) noexcept;
        PavementAnalyzer& operator=(PavementAnalyzer&&) noexcept;
        PavementAnalyzer(const PavementAnalyzer&) = delete;
        PavementAnalyzer& operator=(const PavementAnalyzer&) = delete;

        /**
         * @brief Analyze a section of pavement.
         *
         * @param image Input image.
         * @return Analysis report.
         */
        PavementResult analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::civil