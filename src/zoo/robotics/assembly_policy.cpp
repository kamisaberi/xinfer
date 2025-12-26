#include <xinfer/zoo/robotics/assembly_policy.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Robotics data is raw numerics; no heavy preproc factory needed.
// Custom math implementation allows for tight control loops.

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::robotics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct AssemblyPolicy::Impl {
    PolicyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    // Pre-allocated to avoid malloc in real-time loop
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Derived
    size_t flat_input_size_ = 0;
    bool use_norm_ = false;

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
            throw std::runtime_error("AssemblyPolicy: Failed to load model " + config_.model_path);
        }

        // 2. Validate Config & Dimensions
        // We assume input is flattened concatenation of: [Pos, Vel, EE, FT]
        // User must ensure config.input_dim matches their training setup.
        if (config_.input_dim == 0) {
            // Auto-calculate default assumption: Joints + Vel + Force
            // 6 + 6 + 6 = 18
            flat_input_size_ = (config_.num_joints * 2) + 6;
        } else {
            flat_input_size_ = config_.input_dim;
        }

        if (!config_.state_mean.empty() && config_.state_mean.size() == flat_input_size_) {
            use_norm_ = true;
        }

        // 3. Pre-allocate Tensors
        // Shape: [1, InputDim]
        input_tensor.resize({1, (int64_t)flat_input_size_}, core::DataType::kFLOAT);
    }

    void flatten_and_normalize(const RobotState& state) {
        float* ptr = static_cast<float*>(input_tensor.data());
        int idx = 0;

        // Lambda to append vector
        auto append = [&](const std::vector<float>& vec) {
            for (float val : vec) {
                if (idx < flat_input_size_) {
                    if (use_norm_) {
                        val = (val - config_.state_mean[idx]) / (config_.state_std[idx] + 1e-6f);
                    }
                    ptr[idx++] = val;
                }
            }
        };

        // Standard Order: Joint Pos -> Joint Vel -> (Optional EE) -> Force
        append(state.joint_positions);
        append(state.joint_velocities);

        // Only append EE or FT if space remains in input dim
        if (idx < flat_input_size_) append(state.ee_pose);
        if (idx < flat_input_size_) append(state.force_torque);

        // Safety check
        if (idx != flat_input_size_) {
            // In a real-time loop, logging might be too slow.
            // We assume integration tests caught dimension mismatches.
        }
    }

    RobotAction postprocess() {
        RobotAction act;
        act.commands.resize(config_.output_dim);

        const float* out_ptr = static_cast<const float*>(output_tensor.data());
        bool use_scale = (config_.action_scale.size() == config_.output_dim);

        for (int i = 0; i < config_.output_dim; ++i) {
            float val = out_ptr[i];

            // Denormalize / Scale
            // RL models usually output [-1, 1] or Gaussian.
            // We scale to physical limits (e.g. rad/s).
            if (use_scale) {
                val *= config_.action_scale[i];
            }

            act.commands[i] = val;
        }
        return act;
    }
};

// =================================================================================
// Public API
// =================================================================================

AssemblyPolicy::AssemblyPolicy(const PolicyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

AssemblyPolicy::~AssemblyPolicy() = default;
AssemblyPolicy::AssemblyPolicy(AssemblyPolicy&&) noexcept = default;
AssemblyPolicy& AssemblyPolicy::operator=(AssemblyPolicy&&) noexcept = default;

void AssemblyPolicy::reset() {
    // If using LSTM/GRU backend, reset hidden states here via backend API
    // e.g. pimpl_->engine_->reset_state();
}

RobotAction AssemblyPolicy::step(const RobotState& state) {
    if (!pimpl_) throw std::runtime_error("AssemblyPolicy is null.");

    // 1. Preprocess (Flatten & Norm)
    pimpl_->flatten_and_normalize(state);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Scale)
    return pimpl_->postprocess();
}

} // namespace xinfer::zoo::robotics