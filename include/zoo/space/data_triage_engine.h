#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::space {

    enum class TriagePriority {
        DISCARD = 0,    // Junk data (e.g., Clouds, Noise)
        LOW = 1,        // Bulk data, send eventually
        HIGH = 2,       // Scientific interest (send next pass)
        CRITICAL = 3    // Mission critical / Alert (send NOW)
    };

    /**
     * @brief Result of Triage Analysis.
     */
    struct TriageResult {
        TriagePriority priority;
        float interest_score;     // 0.0 - 1.0 (How valuable is this data?)
        std::string classification; // e.g. "Cloud", "Ship", "Fire"
        float cloud_cover_pct;    // Specific metric for EO satellites
    };

    struct TriageConfig {
        // Hardware Target (Space-grade FPGAs or Myriad/Loihi)
        xinfer::Target target = xinfer::Target::AMD_VITIS;

        // Model Path (e.g., cloud_segmenter.xmodel, ship_detector.engine)
        std::string model_path;

        // Input Specs
        int input_width = 512;
        int input_height = 512;

        // Triage Rules
        float max_cloud_cover = 0.70f; // Discard if > 70% clouds
        float interest_threshold = 0.6f; // Mark HIGH if score > 0.6

        // Vendor flags (e.g. "RAD_TOLERANT=TRUE")
        std::vector<std::string> vendor_params;
    };

    class DataTriageEngine {
    public:
        explicit DataTriageEngine(const TriageConfig& config);
        ~DataTriageEngine();

        // Move semantics
        DataTriageEngine(DataTriageEngine&&) noexcept;
        DataTriageEngine& operator=(DataTriageEngine&&) noexcept;
        DataTriageEngine(const DataTriageEngine&) = delete;
        DataTriageEngine& operator=(const DataTriageEngine&) = delete;

        /**
         * @brief Assess a sensor frame for downlink priority.
         *
         * Pipeline:
         * 1. Check basic signal quality (Histogram).
         * 2. Inference (Cloud Segmentation or Object Detection).
         * 3. Apply Triage Rules.
         *
         * @param image Earth Observation (EO) image frame.
         * @return Priority decision.
         */
        TriageResult evaluate(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::space