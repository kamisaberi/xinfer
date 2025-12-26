#include <xinfer/zoo/vision/smoke_flame_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SmokeFlameDetector::Impl {
    SmokeFlameConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const SmokeFlameConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SmokeFlameDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Postprocessor (YOLO Logic)
        postproc_ = postproc::create_detection(config_.target);

        postproc::DetectionConfig post_cfg;
        // Use the lower of the two thresholds to catch everything initially
        post_cfg.conf_threshold = std::min(config_.fire_thresh, config_.smoke_thresh);
        post_cfg.nms_threshold = config_.nms_threshold;
        postproc_->init(post_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

SmokeFlameDetector::SmokeFlameDetector(const SmokeFlameConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SmokeFlameDetector::~SmokeFlameDetector() = default;
SmokeFlameDetector::SmokeFlameDetector(SmokeFlameDetector&&) noexcept = default;
SmokeFlameDetector& SmokeFlameDetector::operator=(SmokeFlameDetector&&) noexcept = default;

std::vector<HazardResult> SmokeFlameDetector::detect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SmokeFlameDetector is null.");

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
    auto raw_detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    // 4. Filter & Scale
    std::vector<HazardResult> hazards;
    hazards.reserve(raw_detections.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : raw_detections) {
        HazardType type = HazardType::UNKNOWN;
        float thresh = 1.0f;

        // Classify based on Config
        if (det.class_id == pimpl_->config_.class_id_fire) {
            type = HazardType::FIRE;
            thresh = pimpl_->config_.fire_thresh;
        } else if (det.class_id == pimpl_->config_.class_id_smoke) {
            type = HazardType::SMOKE;
            thresh = pimpl_->config_.smoke_thresh;
        } else {
            continue; // Ignore other classes if model has them
        }

        // Apply specific threshold
        if (det.confidence < thresh) continue;

        HazardResult res;
        res.type = type;
        res.confidence = det.confidence;

        // Scale Box
        res.box.x1 = det.x1 * scale_x;
        res.box.y1 = det.y1 * scale_y;
        res.box.x2 = det.x2 * scale_x;
        res.box.y2 = det.y2 * scale_y;
        res.box.class_id = det.class_id;

        hazards.push_back(res);
    }

    return hazards;
}

void SmokeFlameDetector::draw_alerts(cv::Mat& image, const std::vector<HazardResult>& results) {
    for (const auto& h : results) {
        cv::Scalar color;
        std::string label;

        if (h.type == HazardType::FIRE) {
            color = cv::Scalar(0, 0, 255); // Red
            label = "FIRE";
        } else {
            color = cv::Scalar(128, 128, 128); // Grey
            label = "SMOKE";
        }

        cv::Rect rect((int)h.box.x1, (int)h.box.y1,
                      (int)(h.box.x2 - h.box.x1), (int)(h.box.y2 - h.box.y1));

        cv::rectangle(image, rect, color, 2);

        // Label background
        std::string text = label + " " + std::to_string((int)(h.confidence * 100)) + "%";
        int baseline;
        cv::Size textSize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &baseline);

        cv::rectangle(image,
                      cv::Point(rect.x, rect.y - textSize.height - 5),
                      cv::Point(rect.x + textSize.width, rect.y),
                      color, -1);

        cv::putText(image, text, cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    }
}

} // namespace xinfer::zoo::vision