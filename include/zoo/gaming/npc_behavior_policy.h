#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>
#include <include/zoo/rl/policy.h>

namespace xinfer::zoo::gaming {

    struct NPCBehaviorPolicyConfig {
        std::string engine_path;
    };

    class NPCBehaviorPolicy {
    public:
        explicit NPCBehaviorPolicy(const NPCBehaviorPolicyConfig& config);
        ~NPCBehaviorPolicy();

        NPCBehaviorPolicy(const NPCBehaviorPolicy&) = delete;
        NPCBehaviorPolicy& operator=(const NPCBehaviorPolicy&) = delete;
        NPCBehaviorPolicy(NPCBehaviorPolicy&&) noexcept;
        NPCBehaviorPolicy& operator=(NPCBehaviorPolicy&&) noexcept;

        core::Tensor predict_batch(const core::Tensor& npc_state_batch);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::gaming

