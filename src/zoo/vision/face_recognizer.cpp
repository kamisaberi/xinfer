#include <xinfer/zoo/vision/face_recognizer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not needed; embedding normalization logic is simple CPU math here.

#include <iostream>
#include <cmath>
#include <numeric>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FaceRecognizer::Impl {
    FaceRecognizerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const FaceRecognizerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("FaceRecognizer: Failed to load model " + config_.model_path);
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

    // Helper: L2 Normalize vector in-place
    void l2_normalize(std::vector<float>& vec) {
        float sum_sq = 0.0f;
        for (float val : vec) sum_sq += val * val;

        float inv_norm = 1.0f / (std::sqrt(sum_sq) + 1e-10f);
        for (float& val : vec) val *= inv_norm;
    }
};

// =================================================================================
// Public API
// =================================================================================

FaceRecognizer::FaceRecognizer(const FaceRecognizerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FaceRecognizer::~FaceRecognizer() = default;
FaceRecognizer::FaceRecognizer(FaceRecognizer&&) noexcept = default;
FaceRecognizer& FaceRecognizer::operator=(FaceRecognizer&&) noexcept = default;

std::vector<float> FaceRecognizer::get_embedding(const cv::Mat& face_image) {
    if (!pimpl_) throw std::runtime_error("FaceRecognizer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = face_image.data;
    frame.width = face_image.cols;
    frame.height = face_image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Extract & Normalize
    // Output tensor shape is typically [1, 512]
    size_t count = pimpl_->output_tensor.size();
    const float* raw_ptr = static_cast<const float*>(pimpl_->output_tensor.data());

    std::vector<float> embedding(raw_ptr, raw_ptr + count);

    // Most frameworks export ArcFace without the final L2Norm layer to save compute,
    // so we do it on CPU.
    pimpl_->l2_normalize(embedding);

    return embedding;
}

float FaceRecognizer::compute_similarity(const std::vector<float>& emb1,
                                         const std::vector<float>& emb2) {
    if (emb1.size() != emb2.size() || emb1.empty()) return 0.0f;

    // Dot Product (Cosine Similarity if vectors are L2 normalized)
    float dot = 0.0f;
    for (size_t i = 0; i < emb1.size(); ++i) {
        dot += emb1[i] * emb2[i];
    }
    return dot;
}

} // namespace xinfer::zoo::vision