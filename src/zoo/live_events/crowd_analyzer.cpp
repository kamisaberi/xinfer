#include <xinfer/zoo/live_events/crowd_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; density map integration is custom.

#include <iostream>
#include <numeric>
#include <algorithm>

namespace xinfer::zoo::live_events {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CrowdAnalyzer::Impl {
    CrowdConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor; // Density Map

    Impl(const CrowdConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CrowdAnalyzer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // Crowd models often use standard ImageNet stats
        pre_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}};
        preproc_->init(pre_cfg);
    }

    // --- Core Logic: Integrate Density Map ---
    CrowdResult postprocess(const core::Tensor& density_map_tensor, const cv::Size& orig_size) {
        CrowdResult result;

        // 1. Calculate Count
        // Sum all values in the tensor
        const float* ptr = static_cast<const float*>(density_map_tensor.data());
        size_t count = density_map_tensor.size();

        // Simple sum (could use std::accumulate or optimized reduction)
        double sum = 0.0;
        for (size_t i = 0; i < count; ++i) sum += ptr[i];

        result.estimated_count = (int)std::round(sum * config_.density_map_factor);

        // 2. Calculate Density
        if (config_.square_meters_in_view > 0) {
            result.average_density = (float)result.estimated_count / config_.square_meters_in_view;
        }

        // 3. Determine Risk Level
        if (result.average_density > config_.high_density_thresh) {
            result.risk_level = CrowdRiskLevel::HIGH;
        } else if (result.average_density > config_.medium_density_thresh) {
            result.risk_level = CrowdRiskLevel::MEDIUM;
        } else {
            result.risk_level = CrowdRiskLevel::LOW;
        }

        // 4. Create Visualization
        // Wrap tensor as cv::Mat
        auto shape = density_map_tensor.shape();
        int h = shape[2];
        int w = shape[3];
        cv::Mat map(h, w, CV_32F, const_cast<float*>(ptr));

        // Normalize 0-255 for colormap
        cv::Mat map_norm;
        cv::normalize(map, map_norm, 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::Mat map_color;
        cv::applyColorMap(map_norm, map_color, cv::COLORMAP_JET);

        // Resize to original
        cv::resize(map_color, result.density_map, orig_size, 0, 0, cv::INTER_LINEAR);

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

CrowdAnalyzer::CrowdAnalyzer(const CrowdConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CrowdAnalyzer::~CrowdAnalyzer() = default;
CrowdAnalyzer::CrowdAnalyzer(CrowdAnalyzer&&) noexcept = default;
CrowdAnalyzer& CrowdAnalyzer::operator=(CrowdAnalyzer&&) noexcept = default;

CrowdResult CrowdAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("CrowdAnalyzer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->postprocess(pimpl_->output_tensor, image.size());
}

} // namespace xinfer::zoo::live_events