#include <include/zoo/rl/policy.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>

namespace xinfer::zoo::rl {

    struct Policy::Impl {
        PolicyConfig config_;
        std::unique_ptr<core::InferenceEngine> engine_;
    };

    Policy::Policy(const PolicyConfig& config)
        : pimpl_(new Impl{config})
    {
        if (!std::ifstream(pimpl_->config_.engine_path).good()) {
            throw std::runtime_error("RL policy engine file not found: " + pimpl_->config_.engine_path);
        }

        pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);
    }

    Policy::~Policy() = default;
    Policy::Policy(Policy&&) noexcept = default;
    Policy& Policy::operator=(Policy&&) noexcept = default;

    core::Tensor Policy::predict(const core::Tensor& state) {
        if (!pimpl_) throw std::runtime_error("Policy is in a moved-from state.");

        // For a single state, we can treat it as a batch of 1.
        // A more optimized version might handle this differently.
        return predict_batch(state);
    }

    core::Tensor Policy::predict_batch(const core::Tensor& state_batch) {
        if (!pimpl_) throw std::runtime_error("Policy is in a moved-from state.");

        auto output_tensors = pimpl_->engine_->infer({state_batch});

        // A simple policy network typically has one output: the action(s).
        // The ownership of the tensor's GPU memory is moved to the caller.
        return std::move(output_tensors[0]);
    }

} // namespace xinfer::zoo::rl