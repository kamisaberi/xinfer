#include <xinfer/zoo/civil/pavement_analyzer.h>
#include <xinfer/core/logging.h>

// --- We reuse the generic Segmenter module ---
#include <xinfer/zoo/vision/segmenter.h>

#include <iostream>
#include <map>

namespace xinfer::zoo::civil {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PavementAnalyzer::Impl {
    PavementConfig config_;

    // High-level Zoo module for segmentation
    std::unique_ptr<vision::Segmenter> segmenter_;

    // Maps string name to class ID for easier lookup
    std::map<std::string, int> name_to_id_;

    Impl(const PavementConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Initialize the Segmenter
        vision::SegmenterConfig seg_cfg;
        seg_cfg.target = config_.target;
        seg_cfg.model_path = config_.model_path;
        seg_cfg.input_width = config_.input_width;
        seg_cfg.input_height = config_.input_height;

        // Pass class names and colors to the underlying segmenter
        seg_cfg.class_names = config_.class_names;
        seg_cfg.class_colors = config_.class_colors;

        seg_cfg.vendor_params = config_.vendor_params;

        segmenter_ = std::make_unique<vision::Segmenter>(seg_cfg);

        // 2. Build name->id map
        for (size_t i = 0; i < config_.class_names.size(); ++i) {
            name_to_id_[config_.class_names[i]] = i;
        }
    }

    // --- Core Logic: Statistics from Mask ---
    PavementStats calculate_stats(const cv::Mat& mask) {
        PavementStats stats = {0.0f, 0.0f, 0.0f, 1.0f};

        // Count pixels for each class
        std::map<int, int> counts;
        int total_pixels = mask.rows * mask.cols;

        for (int y = 0; y < mask.rows; ++y) {
            for (int x = 0; x < mask.cols; ++x) {
                counts[mask.at<uint8_t>(y, x)]++;
            }
        }

        // Total pavement area (non-background)
        int pavement_pixels = 0;
        for (const auto& kv : counts) {
            if (kv.first != 0) { // Assuming 0 is background
                pavement_pixels += kv.second;
            }
        }
        if (pavement_pixels == 0) pavement_pixels = 1; // Avoid div/0

        // Calculate percentages
        if (name_to_id_.count("Crack")) {
            stats.percent_cracks = (float)counts[name_to_id_["Crack"]] / pavement_pixels;
        }
        if (name_to_id_.count("Pothole")) {
            stats.percent_potholes = (float)counts[name_to_id_["Pothole"]] / pavement_pixels;
        }
        if (name_to_id_.count("Rutting")) {
            stats.percent_rutting = (float)counts[name_to_id_["Rutting"]] / pavement_pixels;
        }

        // Calculate PCI Score (Simple weighted average)
        // Heuristic: Potholes are worse than cracks
        float damage_score = (stats.percent_cracks * 0.5f) +
                             (stats.percent_potholes * 1.0f) +
                             (stats.percent_rutting * 0.7f);

        stats.condition_score = 1.0f - std::min(1.0f, damage_score);

        return stats;
    }
};

// =================================================================================
// Public API
// =================================================================================

PavementAnalyzer::PavementAnalyzer(const PavementConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PavementAnalyzer::~PavementAnalyzer() = default;
PavementAnalyzer::PavementAnalyzer(PavementAnalyzer&&) noexcept = default;
PavementAnalyzer& PavementAnalyzer::operator=(PavementAnalyzer&&) noexcept = default;

PavementResult PavementAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_ || !pimpl_->segmenter_) throw std::runtime_error("PavementAnalyzer is null.");

    // 1. Run Segmentation
    auto seg_res = pimpl_->segmenter_->segment(image);

    PavementResult result;
    result.defect_map = seg_res.color_mask;

    // 2. Calculate Stats from the raw mask
    result.stats = pimpl_->calculate_stats(seg_res.mask);

    return result;
}

} // namespace xinfer::zoo::civil