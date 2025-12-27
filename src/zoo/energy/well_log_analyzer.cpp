#include <xinfer/zoo/energy/well_log_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <algorithm>
#include <cstring>

namespace xinfer::zoo::energy {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct WellLogAnalyzer::Impl {
    WellLogConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const WellLogConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("WellLogAnalyzer: Failed to load model.");
        }

        // 2. Setup Post-processor
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 1;
        post_cfg.apply_softmax = true;
        post_cfg.labels = config_.labels;
        postproc_->init(post_cfg);

        // 3. Pre-allocate Input Tensor
        // Shape: [1, Features, Window] for 1D-CNN
        input_tensor.resize({1, (int64_t)config_.num_features, (int64_t)config_.window_size}, core::DataType::kFLOAT);
    }

    // --- Preprocessing: Flatten & Normalize a Window ---
    void prepare_window(const std::vector<std::vector<float>>& window_data) {
        float* ptr = static_cast<float*>(input_tensor.data());

        // Layout is [Features, Time]
        for (int f = 0; f < config_.num_features; ++f) {
            for (int t = 0; t < config_.window_size; ++t) {
                float val = window_data[t][f];

                // Normalize
                if (f < config_.mean.size()) {
                    val = (val - config_.mean[f]) / (config_.std[f] + 1e-6f);
                }
                ptr[f * config_.window_size + t] = val;
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

WellLogAnalyzer::WellLogAnalyzer(const WellLogConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

WellLogAnalyzer::~WellLogAnalyzer() = default;
WellLogAnalyzer::WellLogAnalyzer(WellLogAnalyzer&&) noexcept = default;
WellLogAnalyzer& WellLogAnalyzer::operator=(WellLogAnalyzer&&) noexcept = default;

LogAnalysisResult WellLogAnalyzer::analyze(const std::map<std::string, std::vector<float>>& log_data,
                                           const std::vector<float>& depth_data) {
    if (!pimpl_) throw std::runtime_error("WellLogAnalyzer is null.");

    LogAnalysisResult result;
    if (log_data.empty() || depth_data.empty()) return result;

    // 1. Collate data into [Time, Features] matrix
    // This is a slow step. In a high-perf system, data would be pre-formatted.
    size_t total_depths = depth_data.size();
    std::vector<std::vector<float>> full_log(total_depths, std::vector<float>(pimpl_->config_.num_features, 0.0f));

    // Assuming the map keys align with the feature order the model expects
    // A robust implementation would map names to indices.
    int feat_idx = 0;
    for (const auto& kv : log_data) {
        if (feat_idx < pimpl_->config_.num_features) {
            for (size_t i = 0; i < total_depths; ++i) {
                full_log[i][feat_idx] = kv.second[i];
            }
            feat_idx++;
        }
    }

    // 2. Sliding Window Inference
    for (size_t i = 0; i + pimpl_->config_.window_size <= total_depths; i += pimpl_->config_.stride) {
        // A. Extract Window
        std::vector<std::vector<float>> window(pimpl_->config_.window_size);
        for(int j=0; j<pimpl_->config_.window_size; ++j) {
            window[j] = full_log[i+j];
        }

        // B. Preprocess
        pimpl_->prepare_window(window);

        // C. Inference
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

        // D. Postprocess
        auto cls_res = pimpl_->postproc_->process(pimpl_->output_tensor);

        if (!cls_res.empty() && !cls_res[0].empty()) {
            auto& top1 = cls_res[0][0];

            LogSegment seg;
            seg.start_depth_m = depth_data[i];
            seg.end_depth_m = depth_data[i + pimpl_->config_.stride - 1]; // Segment covers the stride
            seg.lithofacies = top1.label;
            seg.confidence = top1.score;

            result.segments.push_back(seg);
        }
    }

    // 3. High-level Analysis (Find Reservoirs)
    // Simple heuristic: Find long, contiguous segments of "Sandstone"
    if (!result.segments.empty()) {
        LogSegment current_reservoir = result.segments[0];

        for (size_t i = 1; i < result.segments.size(); ++i) {
            if (result.segments[i].lithofacies == "Sandstone" && current_reservoir.lithofacies == "Sandstone") {
                // Merge
                current_reservoir.end_depth_m = result.segments[i].end_depth_m;
            } else {
                // Flush if it meets criteria
                if (current_reservoir.lithofacies == "Sandstone" &&
                    (current_reservoir.end_depth_m - current_reservoir.start_depth_m) > 5.0f) { // > 5m
                    result.potential_reservoirs.push_back(current_reservoir);
                }
                current_reservoir = result.segments[i];
            }
        }
        // Flush last
        if (current_reservoir.lithofacies == "Sandstone" && (current_reservoir.end_depth_m - current_reservoir.start_depth_m) > 5.0f) {
            result.potential_reservoirs.push_back(current_reservoir);
        }
    }

    return result;
}

} // namespace xinfer::zoo::energy