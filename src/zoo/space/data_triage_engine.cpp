#include <xinfer/zoo/space/data_triage_engine.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <numeric>

namespace xinfer::zoo::space {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DataTriageEngine::Impl {
    TriageConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> seg_postproc_;
    // Could also have a detection postproc if using a detector model

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const TriageConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DataTriageEngine: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Post-processor (Segmentation for Cloud Mask)
        // Assumes model outputs a mask where Class 1 = Cloud
        seg_postproc_ = postproc::create_segmentation(config_.target);

        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.input_width;  // Process at low res for speed
        post_cfg.target_height = config_.input_height;
        seg_postproc_->init(post_cfg);
    }

    // --- Logic: Cloud Cover Calculation ---
    float calculate_cloud_percentage(const core::Tensor& mask_tensor) {
        // Mask is [1, H, W] uint8
        const uint8_t* data = static_cast<const uint8_t*>(mask_tensor.data());
        size_t total_pixels = mask_tensor.size();
        size_t cloud_pixels = 0;

        // Simple count (Assuming Class 1 is Cloud)
        for (size_t i = 0; i < total_pixels; ++i) {
            if (data[i] == 1) {
                cloud_pixels++;
            }
        }
        return (float)cloud_pixels / (float)total_pixels;
    }
};

// =================================================================================
// Public API
// =================================================================================

DataTriageEngine::DataTriageEngine(const TriageConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DataTriageEngine::~DataTriageEngine() = default;
DataTriageEngine::DataTriageEngine(DataTriageEngine&&) noexcept = default;
DataTriageEngine& DataTriageEngine::operator=(DataTriageEngine&&) noexcept = default;

TriageResult DataTriageEngine::evaluate(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DataTriageEngine is null.");

    TriageResult result;
    result.priority = TriagePriority::LOW; // Default
    result.classification = "Unknown";
    result.interest_score = 0.0f;
    result.cloud_cover_pct = 0.0f;

    // 0. Quick Heuristic (Black Image Check)
    // If sensor failed or looking at space, mean intensity will be near 0
    cv::Scalar mean_intensity = cv::mean(image);
    if (mean_intensity[0] < 5.0 && mean_intensity[1] < 5.0 && mean_intensity[2] < 5.0) {
        result.priority = TriagePriority::DISCARD;
        result.classification = "Dark/Empty";
        return result;
    }

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference (Cloud Segmentation)
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto seg_res = pimpl_->seg_postproc_->process(pimpl_->output_tensor);

    // 4. Decision Logic
    float cloud_pct = pimpl_->calculate_cloud_percentage(seg_res.mask);
    result.cloud_cover_pct = cloud_pct;

    if (cloud_pct > pimpl_->config_.max_cloud_cover) {
        result.priority = TriagePriority::DISCARD;
        result.classification = "Cloudy";
        result.interest_score = 0.0f;
    }
    else {
        // Not cloudy. Is it interesting?
        // Inverse of cloud cover is a simple proxy for "visibility"
        result.interest_score = 1.0f - cloud_pct;

        if (result.interest_score > pimpl_->config_.interest_threshold) {
            result.priority = TriagePriority::HIGH;
            result.classification = "Clear/Valuable";
        } else {
            result.priority = TriagePriority::LOW;
            result.classification = "Partial";
        }

        // Future expansion: Run object detection here to upgrade to CRITICAL
        // if (detect_ships(image)) result.priority = CRITICAL;
    }

    return result;
}

} // namespace xinfer::zoo::space