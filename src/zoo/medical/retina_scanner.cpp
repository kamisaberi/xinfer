#include <xinfer/zoo/medical/retina_scanner.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct RetinaScanner::Impl {
    RetinaConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const RetinaConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("RetinaScanner: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = 1.0f;

        preproc_->init(pre_cfg);

        // 3. Setup Classification Post-processor
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 5; // Get all grades
        post_cfg.apply_softmax = true;
        postproc_->init(post_cfg);
    }

    // --- Helper: Crop Black Borders (Fundus ROI) ---
    cv::Mat crop_fundus(const cv::Mat& img) {
        if (!config_.auto_crop_fundus) return img;

        cv::Mat gray;
        if (img.channels() == 3) cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
        else gray = img;

        // Threshold to find the circle
        cv::Mat mask;
        cv::threshold(gray, mask, 10, 255, cv::THRESH_BINARY);

        // Find contours
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (contours.empty()) return img;

        // Find largest contour (the eye)
        double max_area = 0;
        cv::Rect bounding_rect = cv::Rect(0, 0, img.cols, img.rows);

        for (const auto& cnt : contours) {
            double area = cv::contourArea(cnt);
            if (area > max_area) {
                max_area = area;
                bounding_rect = cv::boundingRect(cnt);
            }
        }

        // Ensure rect is valid
        bounding_rect = bounding_rect & cv::Rect(0, 0, img.cols, img.rows);
        if (bounding_rect.width <= 0 || bounding_rect.height <= 0) return img;

        return img(bounding_rect).clone();
    }

    // --- Optional: Ben Graham's Method (Enhance Features) ---
    // Not implemented fully here for brevity, but typically involves:
    // img = img * 4 + 128 - GaussianBlur(img)
    // to normalize lighting conditions.
};

// =================================================================================
// Public API
// =================================================================================

RetinaScanner::RetinaScanner(const RetinaConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

RetinaScanner::~RetinaScanner() = default;
RetinaScanner::RetinaScanner(RetinaScanner&&) noexcept = default;
RetinaScanner& RetinaScanner::operator=(RetinaScanner&&) noexcept = default;

RetinaResult RetinaScanner::scan(const cv::Mat& fundus_image) {
    if (!pimpl_) throw std::runtime_error("RetinaScanner is null.");

    // 1. Smart Crop (Remove black borders)
    cv::Mat roi = pimpl_->crop_fundus(fundus_image);

    // 2. Preprocess
    preproc::ImageFrame frame;
    frame.data = roi.data;
    frame.width = roi.cols;
    frame.height = roi.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 3. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 4. Postprocess
    auto batch_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    RetinaResult result;
    result.refer_to_specialist = false;
    result.confidence = 0.0f;
    result.grade = DRGrade::NO_DR;

    if (!batch_res.empty() && !batch_res[0].empty()) {
        // Assume model outputs classes 0..4 directly
        // Find class with highest probability
        // Since classification postproc sorts by score, index 0 is the best guess

        auto top1 = batch_res[0][0];
        result.confidence = top1.score;
        result.grade = static_cast<DRGrade>(top1.id);

        // Calculate Referral Logic
        // Sum probabilities of Moderate(2), Severe(3), Proliferative(4)
        float referral_prob = 0.0f;
        for (const auto& r : batch_res[0]) {
            if (r.id >= 2) {
                referral_prob += r.score;
            }
        }

        if (referral_prob > pimpl_->config_.referral_threshold) {
            result.refer_to_specialist = true;
        }
    }

    // Optional: Generate GradCAM/Heatmap
    // This requires accessing intermediate layers which simple IBackend doesn't expose easily yet.
    // Placeholder for visual feedback.
    result.attention_map = cv::Mat::zeros(roi.size(), CV_8UC1);

    return result;
}

} // namespace xinfer::zoo::medical