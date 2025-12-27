#include <xinfer/zoo/fashion/pattern_generator.h>
#include <xinfer/core/logging.h>

// --- We reuse the DCGAN module as the underlying engine ---
#include <xinfer/zoo/generative/dcgan.h>

#include <iostream>

namespace xinfer::zoo::fashion {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PatternGenerator::Impl {
    PatternConfig config_;

    // The DCGAN module handles the core GAN logic
    std::unique_ptr<generative::DCGAN> gan_engine_;

    Impl(const PatternConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Configure the underlying DCGAN module
        generative::DcganConfig gan_config;
        gan_config.target = config_.target;
        gan_config.model_path = config_.model_path;
        gan_config.latent_dim = config_.latent_dim;
        gan_config.seed = config_.seed;
        gan_config.vendor_params = config_.vendor_params;

        gan_engine_ = std::make_unique<generative::DCGAN>(gan_config);
    }
};

// =================================================================================
// Public API
// =================================================================================

PatternGenerator::PatternGenerator(const PatternConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PatternGenerator::~PatternGenerator() = default;
PatternGenerator::PatternGenerator(PatternGenerator&&) noexcept = default;
PatternGenerator& PatternGenerator::operator=(PatternGenerator&&) noexcept = default;

cv::Mat PatternGenerator::generate() {
    if (!pimpl_ || !pimpl_->gan_engine_) {
        throw std::runtime_error("PatternGenerator is not initialized.");
    }

    // Delegate the call to the DCGAN module
    return pimpl_->gan_engine_->generate();
}

cv::Mat PatternGenerator::generate_from_vector(const std::vector<float>& latent_vector) {
    if (!pimpl_ || !pimpl_->gan_engine_) {
        throw std::runtime_error("PatternGenerator is not initialized.");
    }

    // Delegate the call
    cv::Mat result = pimpl_->gan_engine_->generate_from_vector(latent_vector);

    // Optional: Post-processing for seamless tiling
    // (e.g., using Poisson blending or specific model architecture features)
    // Here we can do a simple wrap-around blend on the edges.

    // For example, blend left edge with right edge
    // This logic is complex and application-specific, but would go here.
    // cv::Mat left_edge = result.col(0);
    // cv::Mat right_edge = result.col(result.cols - 1);
    // cv::addWeighted(left_edge, 0.5, right_edge, 0.5, 0.0, ...);

    return result;
}

} // namespace xinfer::zoo::fashion