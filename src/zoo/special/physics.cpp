#include <xinfer/zoo/special/physics.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Physics data is raw numerics, so we don't use image/audio preprocessors.
// Custom normalization logic is implemented internally.

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xinfer::zoo::special {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PhysicsEngine::Impl {
    PhysicsConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Normalization Flags
    bool do_normalize_ = false;

    Impl(const PhysicsConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PhysicsEngine: Failed to load model " + config_.model_path);
        }

        // 2. Validate Config
        if (!config_.mean.empty() && config_.mean.size() == config_.input_dim) {
            do_normalize_ = true;
        }
    }

    // --- Helpers ---

    void prepare_batch(const std::vector<PhysicalState>& states) {
        int batch_size = states.size();

        // Resize Input Tensor [Batch, InputDim]
        // Note: Some physics models use [Batch, InputDim, 1], keeping it simple here
        input_tensor.resize({(int64_t)batch_size, (int64_t)config_.input_dim}, core::DataType::kFLOAT);

        float* ptr = static_cast<float*>(input_tensor.data());

        // Flatten and Normalize
        // Parallelize this loop for large batches
        // #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
            const auto& feat = states[b].features;
            float* row = ptr + (b * config_.input_dim);

            for (int i = 0; i < config_.input_dim; ++i) {
                float val = (i < feat.size()) ? feat[i] : 0.0f;

                if (do_normalize_) {
                    // (val - mean) / std
                    float s = config_.std[i];
                    if (std::abs(s) < 1e-9) s = 1.0f;
                    val = (val - config_.mean[i]) / s;
                }
                row[i] = val;
            }
        }
    }

    std::vector<PhysicsResult> process_output(int batch_size) {
        std::vector<PhysicsResult> results(batch_size);

        const float* out_ptr = static_cast<const float*>(output_tensor.data());

        // Check output shape. Assuming [Batch, OutputDim] or [Batch, OutputDim + ClassScore]
        // Let's assume the model outputs Regression (Next State)
        // AND potentially a classification score at the end if OutputDim > InputDim.

        int row_stride = config_.output_dim; // or tensor.shape()[1]

        for (int b = 0; b < batch_size; ++b) {
            const float* row = out_ptr + (b * row_stride);

            results[b].next_state.resize(config_.output_dim);

            // Copy regression values
            // We optionally Denormalize here if the model predicts normalized deltas.
            // Assuming model predicts raw values for this example.
            for (int i = 0; i < config_.output_dim; ++i) {
                results[b].next_state[i] = row[i];
            }

            // Dummy classification logic (e.g. if model output has extra column)
            // results[b].confidence = row[config_.output_dim];
        }

        return results;
    }
};

// =================================================================================
// Public API
// =================================================================================

PhysicsEngine::PhysicsEngine(const PhysicsConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PhysicsEngine::~PhysicsEngine() = default;
PhysicsEngine::PhysicsEngine(PhysicsEngine&&) noexcept = default;
PhysicsEngine& PhysicsEngine::operator=(PhysicsEngine&&) noexcept = default;

PhysicsResult PhysicsEngine::simulate_step(const PhysicalState& current_state) {
    if (!pimpl_) throw std::runtime_error("PhysicsEngine is null.");

    std::vector<PhysicalState> batch = { current_state };
    auto results = simulate_batch(batch);
    return results[0];
}

std::vector<PhysicsResult> PhysicsEngine::simulate_batch(const std::vector<PhysicalState>& states) {
    if (!pimpl_) throw std::runtime_error("PhysicsEngine is null.");
    if (states.empty()) return {};

    // 1. Prepare (Batch Normalize)
    pimpl_->prepare_batch(states);

    // 2. Inference
    // Accelerate huge batches using GPU or FPGA
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    return pimpl_->process_output(states.size());
}

} // namespace xinfer::zoo::special