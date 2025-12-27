#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::energy {

    /**
     * @brief A single classified segment of the well.
     */
    struct LogSegment {
        float start_depth_m;
        float end_depth_m;
        std::string lithofacies; // "Sandstone", "Shale", etc.
        float confidence;
    };

    struct LogAnalysisResult {
        std::vector<LogSegment> segments;

        // Potential pay zones (e.g., contiguous Sandstone > 5m)
        std::vector<LogSegment> potential_reservoirs;
    };

    struct WellLogConfig {
        // Hardware Target (CPU is often sufficient for 1D data)
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., lithofacies_cnn.onnx)
        std::string model_path;

        // Input Specs for the Model
        int window_size = 128; // e.g., 128 depth measurements
        int num_features = 7;  // e.g., GR, NPHI, RHOB, DT, SP, PE, RES

        // Label map (Class ID -> Rock Type)
        std::vector<std::string> labels;

        // Normalization (Mean/Std per feature)
        std::vector<float> mean;
        std::vector<float> std;

        // Stride for the sliding window
        int stride = 64;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class WellLogAnalyzer {
    public:
        explicit WellLogAnalyzer(const WellLogConfig& config);
        ~WellLogAnalyzer();

        // Move semantics
        WellLogAnalyzer(WellLogAnalyzer&&) noexcept;
        WellLogAnalyzer& operator=(WellLogAnalyzer&&) noexcept;
        WellLogAnalyzer(const WellLogAnalyzer&) = delete;
        WellLogAnalyzer& operator=(const WellLogAnalyzer&) = delete;

        /**
         * @brief Analyze a full well log.
         *
         * @param log_data A map of log curve names to their data vectors.
         * @param depth_data Vector of corresponding depth measurements.
         * @return A list of classified rock segments.
         */
        LogAnalysisResult analyze(const std::map<std::string, std::vector<float>>& log_data,
                                  const std::vector<float>& depth_data);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::energy