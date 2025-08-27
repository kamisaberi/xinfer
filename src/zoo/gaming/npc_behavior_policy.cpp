#include <include/zoo/gaming/npc_behavior_policy.h>
#include <stdexcept>
#include <vector>

namespace xinfer::zoo::gaming {

    struct NPCBehaviorPolicy::Impl {
        NPCBehaviorPolicyConfig config_;
        std::unique_ptr<rl::Policy> policy_engine_;
    };

    NPCBehaviorPolicy::NPCBehaviorPolicy(const NPCBehaviorPolicyConfig& config)
        : pimpl_(new Impl{config})
    {
        rl::PolicyConfig policy_config;
        policy_config.engine_path = pimpl_->config_.engine_path;
        pimpl_->policy_engine_ = std::make_unique<rl::Policy>(policy_config);
    }

    NPCBehaviorPolicy::~NPCBehaviorPolicy() = default;
    NPCBehaviorPolicy::NPCBehaviorPolicy(NPCBehaviorPolicy&&) noexcept = default;
    NPCBehaviorPolicy& NPCBehaviorPolicy::operator=(NPCBehaviorPolicy&&) noexcept = default;

    core::Tensor NPCBehaviorPolicy::predict_batch(const core::Tensor& npc_state_batch) {
        if (!pimpl_) throw std::runtime_error("NPCBehaviorPolicy is in a moved-from state.");

        return pimpl_->policy_engine_->predict_batch(npc_state_batch);
    }

} // namespace xinfer::zoo::gaming