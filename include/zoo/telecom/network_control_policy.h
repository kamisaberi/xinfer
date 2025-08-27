#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>
#include <include/zoo/rl/policy.h>

namespace xinfer::zoo::telecom {

    struct NetworkControlPolicyConfig {
        std::string engine_path;
    };

    class NetworkControlPolicy {
    public:
        explicit NetworkControlPolicy(const NetworkControlPolicyConfig& config);
        ~NetworkControlPolicy();

        NetworkControlPolicy(const NetworkControlPolicy&) = delete;
        NetworkControlPolicy& operator=(const NetworkControlPolicy&) = delete;
        NetworkControlPolicy(NetworkControlPolicy&&) noexcept;
        NetworkControlPolicy& operator=(NetworkControlPolicy&&) noexcept;

        core::Tensor predict(const core::Tensor& network_state_tensor);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::telecom

