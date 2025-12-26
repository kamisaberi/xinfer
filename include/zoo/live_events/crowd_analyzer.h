#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::live_events {

    enum class CrowdRiskLevel {
        LOW = 0,    // Sparse crowd
        MEDIUM = 1, // Dense but flowing
        HIGH = 2,   // Crushing risk, no movement
    };

    struct CrowdResult {
        // Core Metrics
        int estimated_count;
        float average_density; // People per square meter
        CrowdRiskLevel risk_level;

        // Visualization
        // Heatmap showing areas of high density
        cv::Mat density_map;
    };

    struct CrowdConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., csrnet.onnx)
        // Expected Output: [1, 1, H, W] Density Heatmap
        std::string model_path;

        // Input Specs
        int input_width = 1024;
        int input_height = 768;

        // Calibration
        // How many "people" does the sum of the density map correspond to?
        // This is a hyperparameter tuned on a validation set.
        float density_map_factor = 2000.0f;

        // Area calibration (for density calculation)
        float square_meters_in_view = 100.0f;

        // Risk Thresholds (People per sq meter)
        float high_density_thresh = 2.0f; // 2 people/m^2
        float medium_density_thresh = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CrowdAnalyzer {
    public:
        explicit CrowdAnalyzer(const CrowdConfig& config);
        ~CrowdAnalyzer();

        // Move semantics
        CrowdAnalyzer(CrowdAnalyzer&&) noexcept;
        CrowdAnalyzer& operator=(CrowdAnalyzer&&) noexcept;
        CrowdAnalyzer(const CrowdAnalyzer&) = delete;
        CrowdAnalyzer& operator=(const CrowdAnalyzer&) = delete;

        /**
         * @brief Analyze a crowd from a camera feed.
         *
         * Pipeline:
         * 1. Preprocess.
         * 2. Inference (Density Map Regression).
         * 3. Postprocess (Integrate map, calculate stats).
         *
         * @param image Input frame.
         * @return Crowd analytics.
         */
        CrowdResult analyze(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::live_events
