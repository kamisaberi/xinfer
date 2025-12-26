#include <xinfer/zoo/telecom/network_control_policy.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc/Postproc factories are skipped for custom numerical logic
// to minimize latency overhead (critical for <1ms loops).

#include <iostream>
#include <deque>
#include <algorithm>
#include <cmath>
#include <numeric>

namespace xinfer::zoo::telecom {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct NetworkController::Impl {
    PolicyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // State History (Rolling Buffer)
    // Flattened history of features
    std::deque<std::vector<float>> history_buffer_;

    Impl(const PolicyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("NetworkController: Failed to load model " + config_.model_path);
        }

        // 2. Validate Config
        // Basic features (4) + Extra
        int expected_feats = 4; // Throughput, Latency, Loss, Queue
        if (config_.input_features < expected_feats) {
             XINFER_LOG_WARN("Config input_features seems low. Ensure it matches model.");
        }

        // 3. Pre-allocate Input Tensor
        // Shape: [1, Lookback, Features] (Standard for LSTM/Transformer RL)
        // Or [1, Lookback * Features] if MLP
        // We assume sequence format: [1, T, F]
        input_tensor.resize({1, (int64_t)config_.lookback_window, (int64_t)config_.input_features}, core::DataType::kFLOAT);

        // Fill buffer with zeros to start
        reset();
    }

    void reset() {
        history_buffer_.clear();
        std::vector<float> zeros(config_.input_features, 0.0f);
        for(int i=0; i<config_.lookback_window; ++i) {
            history_buffer_.push_back(zeros);
        }
    }

    // Convert Struct -> Vector -> Normalize
    std::vector<float> process_state(const NetworkState& s) {
        std::vector<float> raw;
        raw.reserve(config_.input_features);

        // Standard mapping
        raw.push_back(s.throughput_mbps);
        raw.push_back(s.latency_ms);
        raw.push_back(s.packet_loss_rate);
        raw.push_back(s.queue_depth_bytes);

        // Append extras
        if (!s.extra_features.empty()) {
            raw.insert(raw.end(), s.extra_features.begin(), s.extra_features.end());
        }

        // Padding if struct is smaller than config
        while(raw.size() < (size_t)config_.input_features) {
            raw.push_back(0.0f);
        }

        // Normalize
        if (!config_.mean.empty() && !config_.std.empty()) {
            for(size_t i=0; i<raw.size(); ++i) {
                if (i < config_.mean.size()) {
                    raw[i] = (raw[i] - config_.mean[i]) / (config_.std[i] + 1e-6f);
                }
            }
        }

        return raw;
    }

    void update_buffer(const std::vector<float>& norm_feats) {
        history_buffer_.pop_front();
        history_buffer_.push_back(norm_feats);

        // Flatten into Tensor
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;
        for(const auto& vec : history_buffer_) {
            for(float val : vec) {
                ptr[idx++] = val;
            }
        }
    }

    ControlAction decode_output(const core::Tensor& out) {
        ControlAction action;
        const float* data = static_cast<const float*>(out.data());
        int size = out.size(); // Total elements

        if (config_.is_discrete) {
            // ArgMax for Classification / Discrete Action
            int max_idx = 0;
            float max_val = data[0];
            for(int i=1; i<size; ++i) {
                if (data[i] > max_val) {
                    max_val = data[i];
                    max_idx = i;
                }
            }
            action.discrete_action_id = max_idx;
            action.action_confidence = max_val; // Or apply softmax here
        } else {
            // Regression / Continuous Action
            action.discrete_action_id = -1;
            action.action_confidence = 1.0f;
            for(int i=0; i<size; ++i) {
                action.continuous_actions.push_back(data[i]);
            }
        }
        return action;
    }
};

// =================================================================================
// Public API
// =================================================================================

NetworkController::NetworkController(const PolicyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

NetworkController::~NetworkController() = default;
NetworkController::NetworkController(NetworkController&&) noexcept = default;
NetworkController& NetworkController::operator=(NetworkController&&) noexcept = default;

void NetworkController::reset() {
    if (pimpl_) pimpl_->reset();
}

ControlAction NetworkController::step(const NetworkState& state) {
    if (!pimpl_) throw std::runtime_error("NetworkController is null.");

    // 1. Process State (Normalize)
    std::vector<float> norm_feats = pimpl_->process_state(state);

    // 2. Update History Buffer & Tensor
    pimpl_->update_buffer(norm_feats);

    // 3. Inference
    // RL Policies are usually very small (MLP/LSTM), latency is minimal
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 4. Decode Action
    return pimpl_->decode_output(pimpl_->output_tensor);
}

} // namespace xinfer::zoo::telecom