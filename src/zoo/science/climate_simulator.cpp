#include <xinfer/zoo/science/climate_simulator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Note: We skip the generic preproc factory because Climate data is
// scientific float arrays, not 8-bit images. Custom math is used below.

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace xinfer::zoo::science {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ClimateSimulator::Impl {
    ClimateConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Derived Constants
    size_t plane_size_;
    size_t total_elements_;

    Impl(const ClimateConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ClimateSimulator: Failed to load model " + config_.model_path);
        }

        // 2. Compute Dimensions
        int num_vars = config_.variables.size();
        plane_size_ = config_.grid_height * config_.grid_width;
        total_elements_ = num_vars * plane_size_;

        // 3. Pre-allocate Tensors
        // Shape: [1, NumVars, Height, Width]
        std::vector<int64_t> shape = {1, (int64_t)num_vars, (int64_t)config_.grid_height, (int64_t)config_.grid_width};
        input_tensor.resize(shape, core::DataType::kFLOAT);
        // Output shape usually matches input for autoregressive models
        // Some models might output fewer variables, but assuming symmetric here.
    }

    // --- Math: Normalize ---
    // (Value - Mean) / Std
    void prepare_input(const AtmosphericState& state) {
        float* dst = static_cast<float*>(input_tensor.data());
        const float* src = state.data.data();

        if (state.data.size() != total_elements_) {
            XINFER_LOG_ERROR("Input state size mismatch. Expected " + std::to_string(total_elements_));
            return;
        }

        // Parallelize across variables
        int num_vars = config_.variables.size();

        // Ideally use OpenMP here for CPU targets
        for (int v = 0; v < num_vars; ++v) {
            float m = config_.variables[v].mean;
            float s = config_.variables[v].std;
            float inv_s = 1.0f / (s + 1e-9f);

            size_t offset = v * plane_size_;

            // Vectorized loop over grid points
            for (size_t i = 0; i < plane_size_; ++i) {
                dst[offset + i] = (src[offset + i] - m) * inv_s;
            }
        }
    }

    // --- Math: Denormalize ---
    // (Value * Std) + Mean
    AtmosphericState process_output(uint64_t prev_timestamp) {
        AtmosphericState res;
        res.timestamp = prev_timestamp + (config_.step_hours * 3600);
        res.data.resize(total_elements_);

        const float* src = static_cast<const float*>(output_tensor.data());
        float* dst = res.data.data();

        int num_vars = config_.variables.size();

        for (int v = 0; v < num_vars; ++v) {
            float m = config_.variables[v].mean;
            float s = config_.variables[v].std;

            size_t offset = v * plane_size_;

            for (size_t i = 0; i < plane_size_; ++i) {
                dst[offset + i] = (src[offset + i] * s) + m;
            }
        }
        return res;
    }
};

// =================================================================================
// Public API
// =================================================================================

ClimateSimulator::ClimateSimulator(const ClimateConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ClimateSimulator::~ClimateSimulator() = default;
ClimateSimulator::ClimateSimulator(ClimateSimulator&&) noexcept = default;
ClimateSimulator& ClimateSimulator::operator=(ClimateSimulator&&) noexcept = default;

void ClimateSimulator::reset() {
    // Stateless models usually don't need reset, but if using LSTM backend,
    // we would reset hidden states here.
}

AtmosphericState ClimateSimulator::step(const AtmosphericState& current_state) {
    if (!pimpl_) throw std::runtime_error("ClimateSimulator is null.");

    // 1. Normalize
    pimpl_->prepare_input(current_state);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Denormalize
    return pimpl_->process_output(current_state.timestamp);
}

std::vector<AtmosphericState> ClimateSimulator::forecast(const AtmosphericState& initial_state, int num_steps) {
    std::vector<AtmosphericState> trajectory;
    trajectory.reserve(num_steps);

    AtmosphericState current = initial_state;

    // Autoregressive Loop
    for (int i = 0; i < num_steps; ++i) {
        // Run Step
        AtmosphericState next = step(current);

        trajectory.push_back(next);

        // Update for next iteration
        current = next; // Copy (optimization: could use swap/move if careful)
    }

    return trajectory;
}

} // namespace xinfer::zoo::science