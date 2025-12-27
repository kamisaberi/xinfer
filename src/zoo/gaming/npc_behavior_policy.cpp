#include <xinfer/zoo/gaming/npc_behavior_policy.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// No heavy preproc needed, just vector math
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <vector>
#include <algorithm>

namespace xinfer::zoo::gaming {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct NpcBehaviorPolicy::Impl {
    NpcPolicyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    // Reuse Classification Postproc for ArgMax + Softmax
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const NpcPolicyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("NpcBehaviorPolicy: Failed to load model.");
        }

        // 2. Setup Post-processor
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 1;
        post_cfg.apply_softmax = true;
        postproc_->init(post_cfg);

        // 3. Pre-allocate Input
        input_tensor.resize({1, (int64_t)config_.observation_dim}, core::DataType::kFLOAT);
    }

    // --- Core Logic: State -> Tensor ---
    void prepare_input(const NpcObservation& obs) {
        std::vector<float> features;
        features.reserve(config_.observation_dim);

        // Flatten struct into vector
        features.push_back(obs.health_percent);
        features.push_back(obs.has_weapon ? 1.0f : 0.0f);
        features.push_back((float)obs.ammo_count); // Normalize in training!

        features.insert(features.end(), obs.raycast_results.begin(), obs.raycast_results.end());

        features.push_back(obs.is_enemy_visible ? 1.0f : 0.0f);
        features.push_back(obs.enemy_distance);
        features.push_back(obs.enemy_angle_rad);

        // Pad with zeros if observation is smaller than expected
        while (features.size() < (size_t)config_.observation_dim) {
            features.push_back(0.0f);
        }

        // Truncate if too large
        if (features.size() > (size_t)config_.observation_dim) {
            features.resize(config_.observation_dim);
        }

        // Copy to tensor
        std::memcpy(input_tensor.data(), features.data(), features.size() * sizeof(float));
    }
};

// =================================================================================
// Public API
// =================================================================================

NpcBehaviorPolicy::NpcBehaviorPolicy(const NpcPolicyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

NpcBehaviorPolicy::~NpcBehaviorPolicy() = default;
NpcBehaviorPolicy::NpcBehaviorPolicy(NpcBehaviorPolicy&&) noexcept = default;
NpcBehaviorPolicy& NpcBehaviorPolicy::operator=(NpcBehaviorPolicy&&) noexcept = default;

void NpcBehaviorPolicy::reset() {
    // If using RNN/LSTM, reset backend state
    // pimpl_->engine_->reset_state();
}

PolicyResult NpcBehaviorPolicy::get_action(const NpcObservation& obs) {
    if (!pimpl_) throw std::runtime_error("NpcBehaviorPolicy is null.");

    // 1. Prepare Input
    pimpl_->prepare_input(obs);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto results = pimpl_->postproc_->process(pimpl_->output_tensor);

    PolicyResult res;
    if (!results.empty() && !results[0].empty()) {
        res.action = static_cast<NpcAction>(results[0][0].id);
        res.confidence = results[0][0].score;
    } else {
        res.action = NpcAction::IDLE;
        res.confidence = 0.0f;
    }

    return res;
}

} // namespace xinfer::zoo::gaming