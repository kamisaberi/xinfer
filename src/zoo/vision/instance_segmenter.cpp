#include <xinfer/zoo/vision/instance_segmenter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct InstanceSegmenter::Impl {
    InstanceSegmenterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IInstanceSegmentationPostprocessor> postproc_;

    // Data Containers
    // YOLO-Seg has 2 outputs: Detection and Proto
    core::Tensor input_tensor;
    core::Tensor output_det;   // [Batch, 116, 8400]
    core::Tensor output_proto; // [Batch, 32, 160, 160]

    std::vector<std::string> labels_;

    Impl(const InstanceSegmenterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("InstanceSegmenter: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Postprocessor (YOLO-Seg Logic)
        postproc_ = postproc::create_instance_segmentation(config_.target);

        postproc::InstanceSegConfig post_cfg;
        post_cfg.conf_threshold = config_.conf_threshold;
        post_cfg.nms_threshold = config_.nms_threshold;
        post_cfg.target_width = config_.input_width;  // Generate masks at network res
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);

        // 4. Load Labels
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
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

InstanceSegmenter::InstanceSegmenter(const InstanceSegmenterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

InstanceSegmenter::~InstanceSegmenter() = default;
InstanceSegmenter::InstanceSegmenter(InstanceSegmenter&&) noexcept = default;
InstanceSegmenter& InstanceSegmenter::operator=(InstanceSegmenter&&) noexcept = default;

std::vector<InstanceResult> InstanceSegmenter::segment(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("InstanceSegmenter is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    // YOLO-Seg produces two outputs. The order depends on the export.
    // Usually: 0=Detection, 1=Proto.
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_det, pimpl_->output_proto});

    // 3. Postprocess
    auto raw_results = pimpl_->postproc_->process({pimpl_->output_det, pimpl_->output_proto});

    // 4. Format Output & Rescale
    std::vector<InstanceResult> final_results;
    final_results.reserve(raw_results.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& raw : raw_results) {
        InstanceResult res;

        // Scale Box
        res.box.x = (int)(raw.box.x1 * scale_x);
        res.box.y = (int)(raw.box.y1 * scale_y);
        res.box.width = (int)((raw.box.x2 - raw.box.x1) * scale_x);
        res.box.height = (int)((raw.box.y2 - raw.box.y1) * scale_y);

        // Safety clamp
        res.box &= cv::Rect(0, 0, image.cols, image.rows);

        res.confidence = raw.box.confidence;
        res.class_id = raw.box.class_id;

        // Label
        if (res.class_id >= 0 && res.class_id < (int)pimpl_->labels_.size()) {
            res.label = pimpl_->labels_[res.class_id];
        } else {
            res.label = "Class " + std::to_string(res.class_id);
        }

        // Convert Mask Tensor -> cv::Mat
        // Raw mask is usually 640x640 (network size)
        int mh = raw.mask.shape()[1];
        int mw = raw.mask.shape()[2];
        const uint8_t* mdata = static_cast<const uint8_t*>(raw.mask.data());

        cv::Mat network_mask(mh, mw, CV_8U, const_cast<uint8_t*>(mdata));

        // Resize mask to original image size
        // Note: For speed, you might want to skip this and let the user resize
        // only the ROI they care about. But for correctness we do it here.
        cv::resize(network_mask, res.mask, image.size(), 0, 0, cv::INTER_NEAREST);

        final_results.push_back(res);
    }

    return final_results;
}

} // namespace xinfer::zoo::vision