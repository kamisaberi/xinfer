#include <xinfer/zoo/vision/change_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Note: We do custom post-processing logic here for diffing embeddings

#include <iostream>
#include <cmath>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ChangeDetector::Impl {
    ChangeDetectorConfig config_;

    // --- AI Components ---
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    core::Tensor input_tensor; // For current frame
    core::Tensor ref_tensor;   // For reference frame embedding (Cached)
    core::Tensor curr_embedding;

    bool has_reference = false;

    // --- Classical Components ---
    cv::Ptr<cv::BackgroundSubtractor> subtractor_;

    Impl(const ChangeDetectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        if (config_.method == ChangeMethod::AI_SIAMESE) {
            // 1. Initialize AI Backend
            engine_ = backends::BackendFactory::create(config_.target);
            xinfer::Config backend_cfg;
            backend_cfg.model_path = config_.model_path;

            if (!engine_->load_model(backend_cfg.model_path)) {
                throw std::runtime_error("ChangeDetector: Failed to load model " + config_.model_path);
            }

            // 2. Initialize Preprocessor
            preproc_ = preproc::create_image_preprocessor(config_.target);
            preproc::ImagePreprocConfig pre_cfg;
            pre_cfg.target_width = config_.input_width;
            pre_cfg.target_height = config_.input_height;
            pre_cfg.target_format = preproc::ImageFormat::RGB;
            pre_cfg.layout_nchw = true;
            preproc_->init(pre_cfg);

        } else {
            // Initialize Classical Algorithms
            if (config_.method == ChangeMethod::CLASSICAL_KNN) {
                subtractor_ = cv::createBackgroundSubtractorKNN();
            } else {
                subtractor_ = cv::createBackgroundSubtractorMOG2();
            }
        }
    }

    // --- AI Logic: Compute Embedding ---
    void compute_embedding(const cv::Mat& img, core::Tensor& output) {
        preproc::ImageFrame frame;
        frame.data = img.data;
        frame.width = img.cols;
        frame.height = img.rows;
        frame.format = preproc::ImageFormat::BGR;

        preproc_->process(frame, input_tensor);
        engine_->predict({input_tensor}, {output});
    }

    // --- AI Logic: Compare Embeddings ---
    // Calculates Euclidean Distance or Cosine Similarity
    cv::Mat compare_embeddings(const core::Tensor& ref, const core::Tensor& curr) {
        // Assuming output is feature map [1, C, H, W]
        // We calculate pixel-wise distance across channels
        auto shape = ref.shape();
        int channels = shape[1];
        int h = shape[2];
        int w = shape[3];
        int spatial = h * w;

        const float* p_ref = (const float*)ref.data();
        const float* p_curr = (const float*)curr.data();

        std::vector<float> diff_map(spatial, 0.0f);

        // L2 Distance per pixel
        for (int i = 0; i < spatial; ++i) {
            float sum_sq = 0.0f;
            for (int c = 0; c < channels; ++c) {
                float d = p_ref[c * spatial + i] - p_curr[c * spatial + i];
                sum_sq += d * d;
            }
            diff_map[i] = std::sqrt(sum_sq);
        }

        // Convert to CV Mask
        cv::Mat float_diff(h, w, CV_32F, diff_map.data());
        cv::Mat binary_mask;
        // Thresholding the difference
        cv::threshold(float_diff, binary_mask, config_.threshold, 255, cv::THRESH_BINARY);

        cv::Mat uint_mask;
        binary_mask.convertTo(uint_mask, CV_8U);

        // Resize back to original if needed (omitted for brevity, usually done at end)
        return uint_mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

ChangeDetector::ChangeDetector(const ChangeDetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ChangeDetector::~ChangeDetector() = default;
ChangeDetector::ChangeDetector(ChangeDetector&&) noexcept = default;
ChangeDetector& ChangeDetector::operator=(ChangeDetector&&) noexcept = default;

void ChangeDetector::reset() {
    if (!pimpl_) return;
    pimpl_->has_reference = false;
    if (pimpl_->subtractor_) pimpl_->subtractor_->clear();
}

void ChangeDetector::set_reference(const cv::Mat& ref_image) {
    if (!pimpl_) return;

    if (pimpl_->config_.method == ChangeMethod::AI_SIAMESE) {
        pimpl_->compute_embedding(ref_image, pimpl_->ref_tensor);
        pimpl_->has_reference = true;
    } else {
        // For classical, we just "learn" this frame heavily
        pimpl_->subtractor_->apply(ref_image, cv::noArray(), 1.0);
    }
}

ChangeResult ChangeDetector::detect(const cv::Mat& image) {
    if (!pimpl_) return {};

    cv::Mat fg_mask;

    // 1. Generate Foreground Mask
    if (pimpl_->config_.method == ChangeMethod::AI_SIAMESE) {
        if (!pimpl_->has_reference) {
            set_reference(image);
            return {false, 0.0f, cv::Mat::zeros(image.size(), CV_8U), {}};
        }

        pimpl_->compute_embedding(image, pimpl_->curr_embedding);

        // Compare current vs reference
        cv::Mat small_mask = pimpl_->compare_embeddings(pimpl_->ref_tensor, pimpl_->curr_embedding);

        // Resize mask to input image size
        cv::resize(small_mask, fg_mask, image.size(), 0, 0, cv::INTER_NEAREST);

    } else {
        // Classical Method
        pimpl_->subtractor_->apply(image, fg_mask, pimpl_->config_.learning_rate);
    }

    // 2. Cleanup Mask (Morphology)
    // Remove noise dots
    cv::morphologyEx(fg_mask, fg_mask, cv::MORPH_OPEN,
                     cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));

    // 3. Find Contours (Blobs)
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(fg_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    ChangeResult result;
    result.diff_mask = fg_mask;
    result.change_detected = false;

    float total_pixels = (float)(image.cols * image.rows);
    float changed_pixels = 0;

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area > pimpl_->config_.min_area) {
            result.bounding_boxes.push_back(cv::boundingRect(contour));
            changed_pixels += (float)area;
        }
    }

    result.change_ratio = changed_pixels / total_pixels;
    if (!result.bounding_boxes.empty()) {
        result.change_detected = true;
    }

    return result;
}

} // namespace xinfer::zoo::vision