#include <include/zoo/telecom/network_control_policy.h>
#include <stdexcept>
#include <vector>

namespace xinfer::zoo::telecom {

    struct NetworkControlPolicy::Impl {
        NetworkControlPolicyConfig config_;
        std::unique_ptr<rl::Policy> policy_engine_;
    };

    NetworkControlPolicy::NetworkControlPolicy(const NetworkControlPolicyConfig& config)
        : pimpl_(new Impl{config})
    {
        rl::PolicyConfig policy_config;
        policy_config.engine_path = pimpl_->config_.engine_path;
        pimpl_->policy_engine_ = std::make_unique<rl::Policy>(policy_config);
    }

    NetworkControlPolicy::~NetworkControlPolicy() = default;
    NetworkControlPolicy::NetworkControlPolicy(NetworkControlPolicy&&) noexcept = default;
    NetworkControlPolicy& NetworkControlPolicy::operator=(NetworkControlPolicy&&) noexcept = default;

    core::Tensor NetworkControlPolicy::predict(const core::Tensor& network_state_tensor) {
        if (!pimpl_) throw std::runtime_error("NetworkControlPolicy is in a moved-from state.");

        return pimpl_->policy_engine_->predict(network_state_tensor);
    }

} // namespace xinfer::zoo::telecom