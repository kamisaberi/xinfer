#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/core/tensor.h>
#include <include/zoo/rl/policy.h>

namespace xinfer::zoo::drones {

    struct NavigationAction {
        float roll, pitch, yaw, thrust;
    };

    struct NavigationPolicyConfig {
        std::string engine_path;
    };

    class NavigationPolicy {
    public:
        explicit NavigationPolicy(const NavigationPolicyConfig& config);
        ~NavigationPolicy();

        NavigationPolicy(const NavigationPolicy&) = delete;
        NavigationPolicy& operator=(const NavigationPolicy&) = delete;
        NavigationPolicy(NavigationPolicy&&) noexcept;
        NavigationPolicy& operator=(NavigationPolicy&&) noexcept;

        NavigationAction predict(const cv::Mat& depth_image, const std::vector<float>& drone_state);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::drones

