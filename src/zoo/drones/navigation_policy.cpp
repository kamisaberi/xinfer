#include <xinfer/zoo/drones/navigation_policy.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory is not used; action scaling is simple custom math.

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::drones {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct NavigationPolicy::Impl {
    NavPolicyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const NavPolicyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("NavigationPolicy: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // RL policies often use simple 0-1 normalization
        pre_cfg.norm_params.scale_factor = 1.0f / 255.0f;

        preproc_->init(pre_cfg);
    }

    // --- Post-processing: Scale Actions ---
    FlightCommand decode_action(const core::Tensor& tensor) {
        FlightCommand cmd;

        // Output: [1, 4] -> [vx, vy, vz, yaw_rate]
        // Model output is usually Tanh activated -> [-1, 1] range
        const float* ptr = static_cast<const float*>(tensor.data());

        // Scale to physical limits
        cmd.velocity_forward = ptr[0] * config_.max_linear_velocity;
        cmd.velocity_right   = ptr[1] * config_.max_linear_velocity;
        cmd.velocity_up      = ptr[2] * config_.max_linear_velocity;
        cmd.yaw_rate         = ptr[3] * config_.max_angular_velocity;

        return cmd;
    }
};

// =================================================================================
// Public API
// =================================================================================

NavigationPolicy::NavigationPolicy(const NavPolicyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

NavigationPolicy::~NavigationPolicy() = default;
NavigationPolicy::NavigationPolicy(NavigationPolicy&&) noexcept = default;
NavigationPolicy& NavigationPolicy::operator=(NavigationPolicy&&) noexcept = default;

void NavigationPolicy::reset() {
    // Reset RNN state in backend if applicable
    // pimpl_->engine_->reset_state();
}

FlightCommand NavigationPolicy::get_action(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("NavigationPolicy is null.");

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
    return pimpl_->decode_action(pimpl_->output_tensor);
}

} // namespace xinfer::zoo::drones