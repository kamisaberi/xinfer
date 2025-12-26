#include <xinfer/zoo/retail/customer_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <fstream>
#include <map>

namespace xinfer::zoo::retail {

// =================================================================================
// Internal State for Tracking
// =================================================================================
struct TrackedCustomerState {
    CustomerAttributes attrs;
    int frames_since_update = 9999; // Force update immediately on first sight
};

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CustomerAnalyzer::Impl {
    AnalyzerConfig config_;

    // --- Pipeline 1: Detection ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    std::unique_ptr<postproc::ITracker> tracker_;
    core::Tensor det_input, det_output;

    // --- Pipeline 2: Attribute Classification ---
    std::unique_ptr<backends::IBackend> attr_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> attr_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> attr_postproc_;
    core::Tensor attr_input, attr_output;
    std::vector<std::string> attr_labels_;

    // Cache
    std::map<int, TrackedCustomerState> state_cache_;

    Impl(const AnalyzerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.det_model_path;
        d_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("CustomerAnalyzer: Failed to load detector.");
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.det_input_width;
        dp_cfg.target_height = config_.det_input_height;
        dp_cfg.target_format = preproc::ImageFormat::RGB;
        dp_cfg.layout_nchw = true;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = config_.det_conf;
        dpost_cfg.num_classes = 1; // Assuming person-only model
        det_postproc_->init(dpost_cfg);

        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = 30;
        tracker_->init(trk_cfg);

        // 2. Setup Attribute Classifier
        attr_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config a_cfg; a_cfg.model_path = config_.attr_model_path;

        if (!attr_engine_->load_model(a_cfg.model_path)) {
            XINFER_LOG_ERROR("Failed to load attribute model. Analytics will be disabled.");
        }

        attr_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig ap_cfg;
        ap_cfg.target_width = config_.attr_input_width;
        ap_cfg.target_height = config_.attr_input_height;
        ap_cfg.target_format = preproc::ImageFormat::RGB;
        ap_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}}; // ImageNet defaults
        attr_preproc_->init(ap_cfg);

        attr_postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig apost_cfg;
        apost_cfg.top_k = 1;
        load_labels(config_.attr_labels_path);
        apost_cfg.labels = attr_labels_;
        attr_postproc_->init(apost_cfg);
    }

    void load_labels(const std::string& path) {
        if (path.empty()) return;
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            attr_labels_.push_back(line);
        }
    }

    // Helper: Parse classification result into Age/Gender
    // Assumes labels like "Female_18-25" or two separate heads handled by the model.
    // Here we assume a flattened label list for simplicity.
    void parse_attributes(const postproc::ClassResult& res, CustomerAttributes& attrs) {
        // Example Label format: "Male_25-34"
        std::string lbl = res.label;
        size_t underscore = lbl.find('_');
        if (underscore != std::string::npos) {
            attrs.gender = lbl.substr(0, underscore);
            attrs.age_group = lbl.substr(underscore + 1);
        } else {
            attrs.gender = lbl;
            attrs.age_group = "Unknown";
        }
        attrs.confidence = res.score;
        attrs.is_analyzed = true;
    }
};

// =================================================================================
// Public API
// =================================================================================

CustomerAnalyzer::CustomerAnalyzer(const AnalyzerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CustomerAnalyzer::~CustomerAnalyzer() = default;
CustomerAnalyzer::CustomerAnalyzer(CustomerAnalyzer&&) noexcept = default;
CustomerAnalyzer& CustomerAnalyzer::operator=(CustomerAnalyzer&&) noexcept = default;

void CustomerAnalyzer::reset() {
    if (pimpl_) {
        pimpl_->tracker_->reset();
        pimpl_->state_cache_.clear();
    }
}

std::vector<CustomerResult> CustomerAnalyzer::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("CustomerAnalyzer is null.");

    std::vector<CustomerResult> results;

    // --- 1. Detect People ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});
    auto raw_dets = pimpl_->det_postproc_->process({pimpl_->det_output});

    // Scale boxes
    std::vector<postproc::BoundingBox> valid_dets;
    float scale_x = (float)image.cols / pimpl_->config_.det_input_width;
    float scale_y = (float)image.rows / pimpl_->config_.det_input_height;

    for(auto& det : raw_dets) {
        det.x1 *= scale_x; det.x2 *= scale_x;
        det.y1 *= scale_y; det.y2 *= scale_y;
        valid_dets.push_back(det);
    }

    // --- 2. Track People ---
    auto tracks = pimpl_->tracker_->update(valid_dets);

    // --- 3. Analyze Attributes (with Caching) ---
    for (const auto& trk : tracks) {
        CustomerResult res;
        res.track_id = trk.track_id;
        res.box = trk.box;

        // Check Cache
        TrackedCustomerState& state = pimpl_->state_cache_[trk.track_id];

        // Need update?
        if (state.frames_since_update >= pimpl_->config_.attribute_update_interval) {

            // Valid Crop?
            cv::Rect roi(
                (int)std::max(0.0f, trk.box.x1),
                (int)std::max(0.0f, trk.box.y1),
                (int)(trk.box.x2 - trk.box.x1),
                (int)(trk.box.y2 - trk.box.y1)
            );
            roi &= cv::Rect(0, 0, image.cols, image.rows);

            if (roi.width > 20 && roi.height > 20) {
                cv::Mat crop = image(roi);

                // Attribute Inference
                preproc::ImageFrame c_frame{crop.data, crop.cols, crop.rows, preproc::ImageFormat::BGR};
                pimpl_->attr_preproc_->process(c_frame, pimpl_->attr_input);
                pimpl_->attr_engine_->predict({pimpl_->attr_input}, {pimpl_->attr_output});

                // Postprocess
                auto attr_res = pimpl_->attr_postproc_->process(pimpl_->attr_output);

                if (!attr_res.empty() && !attr_res[0].empty()) {
                    pimpl_->parse_attributes(attr_res[0][0], state.attrs);
                    state.frames_since_update = 0;
                }
            }
        } else {
            state.frames_since_update++;
        }

        res.attributes = state.attrs;
        results.push_back(res);
    }

    // Cleanup old cache entries
    // (Simple logic: if cache size > 100, remove IDs not seen recently)
    // In production, sync this with tracker dead tracks.
    if (pimpl_->state_cache_.size() > 200) {
        pimpl_->state_cache_.clear(); // Hard reset for simplicity
    }

    return results;
}

} // namespace xinfer::zoo::retail