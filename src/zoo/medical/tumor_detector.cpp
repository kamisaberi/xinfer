#include <xinfer/zoo/medical/tumor_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TumorDetector::Impl {
    TumorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const TumorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TumorDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;

        // Handle Grayscale vs RGB
        if (config_.input_channels == 1) {
            pre_cfg.target_format = preproc::ImageFormat::GRAY;
        } else {
            pre_cfg.target_format = preproc::ImageFormat::RGB;
        }

        pre_cfg.layout_nchw = true;
        pre_cfg.norm_params.scale_factor = 1.0f / 255.0f; // Basic normalization
        preproc_->init(pre_cfg);

        // 3. Setup Detection Post-processor
        postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig post_cfg;
        post_cfg.conf_threshold = config_.conf_threshold;
        post_cfg.nms_threshold = config_.nms_threshold;

        // If labels provided, set num classes
        if (!config_.labels.empty()) {
            post_cfg.num_classes = config_.labels.size();
        }

        postproc_->init(post_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

TumorDetector::TumorDetector(const TumorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TumorDetector::~TumorDetector() = default;
TumorDetector::TumorDetector(TumorDetector&&) noexcept = default;
TumorDetector& TumorDetector::operator=(TumorDetector&&) noexcept = default;

std::vector<TumorResult> TumorDetector::detect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("TumorDetector is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;

    // Auto-detect format based on OpenCV Mat channels
    if (image.channels() == 1) {
        frame.format = preproc::ImageFormat::GRAY;
    } else {
        frame.format = preproc::ImageFormat::BGR;
    }

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    // 4. Map Results
    std::vector<TumorResult> results;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (auto& det : detections) {
        TumorResult res;

        // Scale Box
        res.box = det;
        res.box.x1 *= scale_x; res.box.x2 *= scale_x;
        res.box.y1 *= scale_y; res.box.y2 *= scale_y;

        res.confidence = det.confidence;
        res.class_id = det.class_id;

        // Label Lookup
        if (det.class_id >= 0 && det.class_id < (int)pimpl_->config_.labels.size()) {
            res.label = pimpl_->config_.labels[det.class_id];
        } else {
            res.label = "Tumor";
        }

        // Logic: Mark as critical if confidence is high or specific class (e.g. Malignant)
        // Here we assume simple confidence threshold for alert
        res.is_critical = (det.confidence > 0.7f);

        results.push_back(res);
    }

    return results;
}

void TumorDetector::visualize(cv::Mat& image, const std::vector<TumorResult>& results) {
    // Ensure we draw on a 3-channel image even if input is grayscale
    if (image.channels() == 1) {
        cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
    }

    for (const auto& res : results) {
        cv::Scalar color = res.is_critical ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 255); // Red or Yellow

        cv::Rect rect(
            (int)res.box.x1, (int)res.box.y1,
            (int)(res.box.x2 - res.box.x1), (int)(res.box.y2 - res.box.y1)
        );

        cv::rectangle(image, rect, color, 2);

        std::string label = res.label + ": " + std::to_string((int)(res.confidence * 100)) + "%";
        cv::putText(image, label, cv::Point(rect.x, rect.y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
    }
}

} // namespace xinfer::zoo::medical