#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>
#include <include/zoo/rl/policy.h>

namespace xinfer::zoo::hft {

    enum class OrderActionType {
        DO_NOTHING,
        PLACE_BUY,
        PLACE_SELL
    };

    struct OrderExecutionAction {
        OrderActionType action;
        float volume;
        float price_level;
    };

    struct OrderExecutionPolicyConfig {
        std::string engine_path;
    };

    class OrderExecutionPolicy {
    public:
        explicit OrderExecutionPolicy(const OrderExecutionPolicyConfig& config);
        ~OrderExecutionPolicy();

        OrderExecutionPolicy(const OrderExecutionPolicy&) = delete;
        OrderExecutionPolicy& operator=(const OrderExecutionPolicy&) = delete;
        OrderExecutionPolicy(OrderExecutionPolicy&&) noexcept;
        OrderExecutionPolicy& operator=(OrderExecutionPolicy&&) noexcept;

        OrderExecutionAction predict(const core::Tensor& market_state_tensor);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::hft

