#pragma once




#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>

namespace xinfer::zoo::rl {

    /**
     * @struct PolicyConfig
     * @brief Configuration for loading a reinforcement learning policy.
     */
    struct PolicyConfig {
        // Path to the pre-built, optimized TensorRT .engine file for the policy network.
        std::string engine_path;
    };

    /**
     * @class Policy
     * @brief A high-level, hyper-optimized engine for executing a trained RL policy.
     *
     * This class loads a pre-built TensorRT engine for a policy network (typically an MLP)
     * and provides a simple, low-latency predict() function to get an action from a state.
     */
    class Policy {
    public:
        explicit Policy(const PolicyConfig& config);
        ~Policy();

        Policy(const Policy&) = delete;
        Policy& operator=(const Policy&) = delete;
        Policy(Policy&&) noexcept;
        Policy& operator=(Policy&&) noexcept;

        /**
         * @brief Gets the best action for a given state observation.
         * @param state A GPU tensor representing the current state of the environment.
         * @return A GPU tensor representing the chosen action.
         */
        core::Tensor predict(const core::Tensor& state);

        /**
         * @brief Gets the best action for a batch of states.
         * @param state_batch A batched GPU tensor of states.
         * @return A batched GPU tensor of the chosen actions.
         */
        core::Tensor predict_batch(const core::Tensor& state_batch);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::rl

