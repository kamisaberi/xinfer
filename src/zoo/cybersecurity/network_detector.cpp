#include <xinfer/zoo/cybersecurity/network_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <iostream>
#include <vector>
#include <cmath>

namespace xinfer::zoo::cybersecurity {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct NetworkDetector::Impl {
    NetworkDetectorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<postproc::IClassificationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Feature count
    int num_features_ = 0;

    Impl(const NetworkDetectorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("NetworkDetector: Failed to load model.");
        }

        // 2. Setup Post-processor
        postproc_ = postproc::create_classification(config_.target);
        postproc::ClassificationConfig post_cfg;
        post_cfg.top_k = 1;
        post_cfg.apply_softmax = true;
        post_cfg.labels = config_.labels;
        postproc_->init(post_cfg);

        // 3. Determine Feature Count
        num_features_ = config_.mean.size();
        if (num_features_ == 0) {
            XINFER_LOG_ERROR("Normalization 'mean' must be provided to determine feature count.");
        }

        // 4. Pre-allocate Tensor
        input_tensor.resize({1, (int64_t)num_features_}, core::DataType::kFLOAT);
    }

    // --- Core Logic: Struct -> Normalized Tensor ---
    void prepare_input(const NetworkFlow& flow) {
        // Flatten struct into a raw float vector
        std::vector<float> features;
        features.reserve(num_features_);

        // This order MUST match the training data
        features.push_back((float)flow.src_port);
        features.push_back((float)flow.dst_port);
        features.push_back((float)flow.flow_duration);
        features.push_back((float)flow.total_fwd_packets);
        features.push_back((float)flow.total_bwd_packets);
        features.push_back((float)flow.total_fwd_bytes);
        features.push_back((float)flow.total_bwd_bytes);
        // ... and ~70 more features for a real IDS model.

        // Protocol One-Hot Encoding (Simplified)
        features.push_back((flow.protocol == "TCP") ? 1.0f : 0.0f);
        features.push_back((flow.protocol == "UDP") ? 1.0f : 0.0f);

        // Pad if necessary
        while(features.size() < (size_t)num_features_) features.push_back(0.0f);

        // Normalize and copy to tensor
        float* ptr = static_cast<float*>(input_tensor.data());
        for(int i=0; i < num_features_; ++i) {
            float val = features[i];
            float s = config_.std[i];
            ptr[i] = (val - config_.mean[i]) / (s + 1e-9f);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

NetworkDetector::NetworkDetector(const NetworkDetectorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

NetworkDetector::~NetworkDetector() = default;
NetworkDetector::NetworkDetector(NetworkDetector&&) noexcept = default;
NetworkDetector& NetworkDetector::operator=(NetworkDetector&&) noexcept = default;

IntrusionResult NetworkDetector::analyze(const NetworkFlow& flow) {
    if (!pimpl_) throw std::runtime_error("NetworkDetector is null.");

    // 1. Prepare Input
    pimpl_->prepare_input(flow);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto results = pimpl_->postproc_->process(pimpl_->output_tensor);

    IntrusionResult res;
    if (!results.empty() && !results[0].empty()) {
        const auto& top1 = results[0][0];
        res.attack_type = top1.label;
        res.confidence = top1.score;

        // Benign traffic is usually labeled "Benign" or is class 0
        res.is_attack = (res.attack_type != "Benign");
    } else {
        res.is_attack = false;
        res.attack_type = "Unknown";
        res.confidence = 0.0f;
    }

    return res;
}

} // namespace xinfer::zoo::cybersecurity