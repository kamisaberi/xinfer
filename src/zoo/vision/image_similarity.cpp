#include <xinfer/zoo/vision/image_similarity.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <cmath>
#include <numeric>
#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ImageSimilarity::Impl {
    ImageSimilarityConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const ImageSimilarityConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ImageSimilarity: Failed to load model " + config_.model_path);
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
        pre_cfg.norm_params.scale_factor = 1.0f; // Input 0-255

        preproc_->init(pre_cfg);
    }

    // Helper: L2 Normalize a vector in-place
    void normalize_vector(std::vector<float>& v) {
        float sum_sq = 0.0f;
        for (float val : v) sum_sq += val * val;

        // Avoid div by zero
        float norm = std::sqrt(sum_sq) + 1e-9f;
        float inv_norm = 1.0f / norm;

        for (float& val : v) val *= inv_norm;
    }
};

// =================================================================================
// Public API
// =================================================================================

ImageSimilarity::ImageSimilarity(const ImageSimilarityConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ImageSimilarity::~ImageSimilarity() = default;
ImageSimilarity::ImageSimilarity(ImageSimilarity&&) noexcept = default;
ImageSimilarity& ImageSimilarity::operator=(ImageSimilarity&&) noexcept = default;

std::vector<float> ImageSimilarity::get_features(const cv::Mat& img) {
    if (!pimpl_) throw std::runtime_error("ImageSimilarity is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = img.data;
    frame.width = img.cols;
    frame.height = img.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Extract & Normalize
    size_t count = pimpl_->output_tensor.size();
    const float* data = static_cast<const float*>(pimpl_->output_tensor.data());

    std::vector<float> features(data, data + count);
    pimpl_->normalize_vector(features);

    return features;
}

float ImageSimilarity::compare(const cv::Mat& img1, const cv::Mat& img2) {
    std::vector<float> feat1 = get_features(img1);
    std::vector<float> feat2 = get_features(img2);

    return compute_cosine_similarity(feat1, feat2);
}

float ImageSimilarity::compute_cosine_similarity(const std::vector<float>& vec_a,
                                                 const std::vector<float>& vec_b) {
    if (vec_a.size() != vec_b.size() || vec_a.empty()) return 0.0f;

    // Assumes vectors are already L2 normalized by get_features()
    // Dot product of normalized vectors == Cosine Similarity
    float dot = 0.0f;
    for (size_t i = 0; i < vec_a.size(); ++i) {
        dot += vec_a[i] * vec_b[i];
    }
    return dot;
}

} // namespace xinfer::zoo::vision