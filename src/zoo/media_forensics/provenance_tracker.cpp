#include <xinfer/zoo/media_forensics/provenance_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// No generic postproc; embedding normalization and search logic is custom.

#include <iostream>
#include <algorithm>
#include <cmath>
#include <map>

namespace xinfer::zoo::media_forensics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ProvenanceTracker::Impl {
    ProvenanceConfig config_;

    // --- Components: Embedder ---
    std::unique_ptr<backends::IBackend> embed_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> embed_preproc_;
    core::Tensor embed_input, embed_output;

    // --- Components: Watermark Decoder (Optional) ---
    std::unique_ptr<backends::IBackend> wm_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> wm_preproc_;
    core::Tensor wm_input, wm_output;

    // --- Database ---
    // ID -> Normalized Feature Vector
    std::map<std::string, std::vector<float>> database_;

    Impl(const ProvenanceConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Embedder
        embed_engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config e_cfg;
        e_cfg.model_path = config_.embedder_path;
        e_cfg.vendor_params = config_.vendor_params;

        if (!embed_engine_->load_model(e_cfg.model_path)) {
            throw std::runtime_error("ProvenanceTracker: Failed to load embedder " + config_.embedder_path);
        }

        embed_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig ep_cfg;
        ep_cfg.target_width = config_.input_width;
        ep_cfg.target_height = config_.input_height;
        ep_cfg.target_format = preproc::ImageFormat::RGB;
        ep_cfg.layout_nchw = true;
        // Standard ImageNet Norm
        ep_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}};
        embed_preproc_->init(ep_cfg);

        // 2. Setup Watermark Decoder (If provided)
        if (!config_.decoder_path.empty()) {
            wm_engine_ = backends::BackendFactory::create(config_.target);
            xinfer::Config w_cfg;
            w_cfg.model_path = config_.decoder_path;

            if (wm_engine_->load_model(w_cfg.model_path)) {
                wm_preproc_ = preproc::create_image_preprocessor(config_.target);
                preproc::ImagePreprocConfig wp_cfg;
                // Watermark decoders often need specific fixed resolution (e.g. 128x128)
                wp_cfg.target_width = 128;
                wp_cfg.target_height = 128;
                wp_cfg.target_format = preproc::ImageFormat::RGB;
                // Usually [-1, 1] scaling
                wp_cfg.norm_params.scale_factor = 1.0f/127.5f;
                wp_cfg.norm_params.mean = {127.5, 127.5, 127.5};
                wm_preproc_->init(wp_cfg);
            }
        }
    }

    // --- Helper: Get Normalized Embedding ---
    std::vector<float> compute_embedding(const cv::Mat& img) {
        // Preprocess
        preproc::ImageFrame frame;
        frame.data = img.data;
        frame.width = img.cols;
        frame.height = img.rows;
        frame.format = preproc::ImageFormat::BGR;

        embed_preproc_->process(frame, embed_input);

        // Inference
        embed_engine_->predict({embed_input}, {embed_output});

        // Normalize (L2)
        size_t count = embed_output.size();
        const float* ptr = static_cast<const float*>(embed_output.data());

        std::vector<float> vec(ptr, ptr + count);

        float sum_sq = 0.0f;
        for (float v : vec) sum_sq += v * v;
        float inv_norm = 1.0f / (std::sqrt(sum_sq) + 1e-9f);
        for (float& v : vec) v *= inv_norm;

        return vec;
    }

    // --- Helper: Decode Watermark ---
    std::string decode_watermark(const cv::Mat& img) {
        if (!wm_engine_) return "";

        preproc::ImageFrame frame;
        frame.data = img.data;
        frame.width = img.cols;
        frame.height = img.rows;
        frame.format = preproc::ImageFormat::BGR;

        wm_preproc_->process(frame, wm_input);
        wm_engine_->predict({wm_input}, {wm_output});

        // Assume output is binary logits [1, NumBits]
        const float* ptr = static_cast<const float*>(wm_output.data());
        size_t bits = wm_output.size();

        std::string payload = "";
        for(size_t i=0; i<bits; ++i) {
            payload += (ptr[i] > 0.0f) ? "1" : "0";
        }
        return payload;
    }
};

// =================================================================================
// Public API
// =================================================================================

ProvenanceTracker::ProvenanceTracker(const ProvenanceConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ProvenanceTracker::~ProvenanceTracker() = default;
ProvenanceTracker::ProvenanceTracker(ProvenanceTracker&&) noexcept = default;
ProvenanceTracker& ProvenanceTracker::operator=(ProvenanceTracker&&) noexcept = default;

void ProvenanceTracker::clear_database() {
    if (pimpl_) pimpl_->database_.clear();
}

void ProvenanceTracker::register_asset(const std::string& id, const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ProvenanceTracker is null.");

    auto vec = pimpl_->compute_embedding(image);
    pimpl_->database_[id] = vec;

    XINFER_LOG_INFO("Registered asset: " + id);
}

ProvenanceResult ProvenanceTracker::trace(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("ProvenanceTracker is null.");

    ProvenanceResult result;
    result.is_known_content = false;
    result.similarity_score = 0.0f;
    result.origin_id = "Unknown";
    result.has_watermark = false;

    // 1. Calculate Fingerprint
    auto query_vec = pimpl_->compute_embedding(image);

    // 2. Linear Search Database (Cosine Similarity)
    // Note: Vectors are already L2 normalized, so Dot Product == Cosine Sim.
    float max_score = -1.0f;
    std::string best_match_id;

    for (const auto& kv : pimpl_->database_) {
        const auto& target_vec = kv.second;

        // Simple Dot Product
        float dot = 0.0f;
        for(size_t i=0; i<query_vec.size(); ++i) {
            dot += query_vec[i] * target_vec[i];
        }

        if (dot > max_score) {
            max_score = dot;
            best_match_id = kv.first;
        }
    }

    result.similarity_score = max_score;
    if (max_score > pimpl_->config_.match_threshold) {
        result.is_known_content = true;
        result.origin_id = best_match_id;
    }

    // 3. Watermark Check (Optional)
    if (pimpl_->wm_engine_) {
        result.watermark_payload = pimpl_->decode_watermark(image);
        if (!result.watermark_payload.empty()) {
            result.has_watermark = true;
        }
    }

    return result;
}

} // namespace xinfer::zoo::media_forensics