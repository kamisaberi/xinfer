#include <include/zoo/drones/navigation_policy.h>
#include <stdexcept>
#include <vector>

namespace xinfer::zoo::drones {

    struct NavigationPolicy::Impl {
        NavigationPolicyConfig config_;
        std::unique_ptr<rl::Policy> policy_engine_;
        // This would also contain a preprocessor for the depth image
    };

    NavigationPolicy::NavigationPolicy(const NavigationPolicyConfig& config)
        : pimpl_(new Impl{config})
    {
        rl::PolicyConfig policy_config;
        policy_config.engine_path = pimpl_->config_.engine_path;
        pimpl_->policy_engine_ = std::make_unique<rl::Policy>(policy_config);
    }

    NavigationPolicy::~NavigationPolicy() = default;
    NavigationPolicy::NavigationPolicy(NavigationPolicy&&) noexcept = default;
    NavigationPolicy& NavigationPolicy::operator=(NavigationPolicy&&) noexcept = default;

    NavigationAction NavigationPolicy::predict(const cv::Mat& depth_image, const std::vector<float>& drone_state) {
        if (!pimpl_) throw std::runtime_error("NavigationPolicy is in a moved-from state.");

        // Placeholder for a real implementation:
        // 1. Pre-process depth_image and drone_state into a single state tensor.
        // 2. Run policy_engine_.predict() on the state tensor.
        // 3. Copy the output action tensor to a std::vector.
        // 4. Populate and return the NavigationAction struct.

        NavigationAction action = {0.0f, 0.0f, 0.0f, 0.0f};
        return action;
    }

} // namespace xinfer::zoo::drones