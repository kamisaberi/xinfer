#include <include/zoo/hft/order_execution_policy.h>
#include <stdexcept>
#include <vector>

namespace xinfer::zoo::hft {

    struct OrderExecutionPolicy::Impl {
        OrderExecutionPolicyConfig config_;
        std::unique_ptr<rl::Policy> policy_engine_;
    };

    OrderExecutionPolicy::OrderExecutionPolicy(const OrderExecutionPolicyConfig& config)
        : pimpl_(new Impl{config})
    {
        rl::PolicyConfig policy_config;
        policy_config.engine_path = pimpl_->config_.engine_path;
        pimpl_->policy_engine_ = std::make_unique<rl::Policy>(policy_config);
    }

    OrderExecutionPolicy::~OrderExecutionPolicy() = default;
    OrderExecutionPolicy::OrderExecutionPolicy(OrderExecutionPolicy&&) noexcept = default;
    OrderExecutionPolicy& OrderExecutionPolicy::operator=(OrderExecutionPolicy&&) noexcept = default;

    OrderExecutionAction OrderExecutionPolicy::predict(const core::Tensor& market_state_tensor) {
        if (!pimpl_) throw std::runtime_error("OrderExecutionPolicy is in a moved-from state.");

        core::Tensor action_tensor = pimpl_->policy_engine_->predict(market_state_tensor);

        std::vector<float> action_vec(action_tensor.num_elements());
        action_tensor.copy_to_host(action_vec.data());

        OrderExecutionAction final_action = {OrderActionType::DO_NOTHING, 0.0f, 0.0f};
        if (action_vec.size() >= 3) {
            final_action.action = static_cast<OrderActionType>(static_cast<int>(action_vec[0]));
            final_action.volume = action_vec[1];
            final_action.price_level = action_vec[2];
        }

        return final_action;
    }

} // namespace xinfer::zoo::hft