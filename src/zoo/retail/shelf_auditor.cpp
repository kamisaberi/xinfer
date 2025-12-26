#include <xinfer/zoo/retail/shelf_auditor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::retail {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ShelfAuditor::Impl {
    AuditorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // SKU Labels
    std::vector<std::string> labels_;

    Impl(const AuditorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ShelfAuditor: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Post-processor
        postproc_ = postproc::create_detection(config_.target);

        postproc::DetectionConfig post_cfg;
        post_cfg.conf_threshold = config_.conf_threshold;
        post_cfg.nms_threshold = config_.nms_threshold;
        postproc_->init(post_cfg);

        // 4. Load Labels
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            XINFER_LOG_WARN("Could not open labels file: " + path);
            return;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

ShelfAuditor::ShelfAuditor(const AuditorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ShelfAuditor::~ShelfAuditor() = default;
ShelfAuditor::ShelfAuditor(ShelfAuditor&&) noexcept = default;
ShelfAuditor& ShelfAuditor::operator=(ShelfAuditor&&) noexcept = default;

ShelfState ShelfAuditor::audit(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ShelfAuditor is null.");

    ShelfState state;

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
    auto detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    // 4. Organize Results
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : detections) {
        // Rescale Box
        postproc::BoundingBox box = det;
        box.x1 *= scale_x; box.x2 *= scale_x;
        box.y1 *= scale_y; box.y2 *= scale_y;

        // Is this a Gap or a Product?
        if (det.class_id == pimpl_->config_.gap_class_id) {
            state.gaps.push_back(box);
        } else {
            ProductItem item;
            item.box = box;
            item.class_id = det.class_id;
            item.confidence = det.confidence;

            if (det.class_id >= 0 && det.class_id < (int)pimpl_->labels_.size()) {
                item.sku_name = pimpl_->labels_[det.class_id];
            } else {
                item.sku_name = "SKU_" + std::to_string(det.class_id);
            }

            state.items.push_back(item);
            state.inventory_summary[item.sku_name]++;
        }
    }

    return state;
}

} // namespace xinfer::zoo::retail