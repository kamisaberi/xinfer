#include <xinfer/zoo/science/transient_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::science {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct TransientDetector::Impl {
    TransientConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // State
    cv::Mat reference_image_;
    bool has_reference_ = false;

    Impl(const TransientConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend (Classifier)
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("TransientDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor (For the small crops)
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.crop_size;
        pre_cfg.target_height = config_.crop_size;
        pre_cfg.target_format = preproc::ImageFormat::GRAY; // Astronomy usually Mono
        pre_cfg.layout_nchw = true;
        pre_cfg.norm_params.scale_factor = 1.0f / 255.0f; // Scale 0-1

        preproc_->init(pre_cfg);

        // 3. Setup Post-processor (Classification)
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig cls_cfg;
        cls_cfg.top_k = 1;
        // Assume Class 1 = "Real", Class 0 = "Bogus"
        postproc_->init(cls_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

TransientDetector::TransientDetector(const TransientConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

TransientDetector::~TransientDetector() = default;
TransientDetector::TransientDetector(TransientDetector&&) noexcept = default;
TransientDetector& TransientDetector::operator=(TransientDetector&&) noexcept = default;

void TransientDetector::set_reference(const cv::Mat& ref_image) {
    // Clone to ensure data persistence
    // Ensure grayscale
    if (ref_image.channels() == 3) {
        cv::cvtColor(ref_image, pimpl_->reference_image_, cv::COLOR_BGR2GRAY);
    } else {
        pimpl_->reference_image_ = ref_image.clone();
    }
    pimpl_->has_reference = true;
}

std::vector<TransientEvent> TransientDetector::detect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("TransientDetector is null.");

    // 0. Prepare Input
    cv::Mat gray_img;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray_img, cv::COLOR_BGR2GRAY);
    } else {
        gray_img = image; // No copy if possible
    }

    if (!pimpl_->has_reference_) {
        // Auto-set reference on first frame
        set_reference(gray_img);
        return {};
    }

    if (gray_img.size() != pimpl_->reference_image_.size()) {
        XINFER_LOG_ERROR("Input image size does not match reference image.");
        return {};
    }

    // 1. Classical Difference Imaging (CPU/OpenCV)
    // Diff = |Science - Reference|
    cv::Mat diff;
    cv::absdiff(gray_img, pimpl_->reference_image_, diff);

    // 2. Candidate Extraction
    cv::Mat thresh;
    cv::threshold(diff, thresh, pimpl_->config_.diff_threshold, 255, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<TransientEvent> confirmed_events;

    // 3. Candidate Validation (Batching loop for ML)
    // Note: For maximum speed, we should collect all crops into a single Batch Tensor.
    // For simplicity here, we process one-by-one.

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < pimpl_->config_.min_area) continue;

        cv::Rect box = cv::boundingRect(contour);

        // Extract ROI (centered and padded)
        int cx = box.x + box.width / 2;
        int cy = box.y + box.height / 2;
        int half_size = pimpl_->config_.crop_size / 2;

        int x1 = std::max(0, cx - half_size);
        int y1 = std::max(0, cy - half_size);
        int x2 = std::min(diff.cols, cx + half_size);
        int y2 = std::min(diff.rows, cy + half_size);

        if (x2 - x1 < 5 || y2 - y1 < 5) continue; // Too small crop

        cv::Mat crop = diff(cv::Rect(x1, y1, x2 - x1, y2 - y1));

        // A. Preprocess Crop
        preproc::ImageFrame crop_frame;
        crop_frame.data = crop.data;
        crop_frame.width = crop.cols;
        crop_frame.height = crop.rows;
        crop_frame.format = preproc::ImageFormat::GRAY;

        pimpl_->preproc_->process(crop_frame, pimpl_->input_tensor);

        // B. Inference ("Real-Bogus" Check)
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

        // C. Postprocess
        auto results = pimpl_->postproc_->process(pimpl_->output_tensor);

        // Assume Class 1 is "Real"
        // Results is [Batch][TopK]
        if (!results.empty() && !results[0].empty()) {
            auto& res = results[0][0];
            if (res.id == 1 && res.score > pimpl_->config_.ml_confidence_thresh) {

                // Calculate Flux (Sum of pixel values in difference map)
                cv::Scalar total_flux = cv::sum(crop);

                TransientEvent evt;
                evt.x = (float)cx;
                evt.y = (float)cy;
                evt.flux = (float)total_flux[0];
                evt.confidence = res.score;
                evt.box = {(float)box.x, (float)box.y, (float)(box.x+box.width), (float)(box.y+box.height), res.score, 1};

                confirmed_events.push_back(evt);
            }
        }
    }

    return confirmed_events;
}

} // namespace xinfer::zoo::science