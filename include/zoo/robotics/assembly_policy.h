#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/core/tensor.h>
#include <include/zoo/rl/policy.h>
#include <include/zoo/vision/classifier.h> // Example for a vision encoder

namespace xinfer::zoo::robotics {

    struct AssemblyPolicyConfig {
        std::string vision_encoder_engine_path;
        std::string policy_engine_path;
    };

    class AssemblyPolicy {
    public:
        explicit AssemblyPolicy(const AssemblyPolicyConfig& config);
        ~AssemblyPolicy();

        AssemblyPolicy(const AssemblyPolicy&) = delete;
        AssemblyPolicy& operator=(const AssemblyPolicy&) = delete;
        AssemblyPolicy(AssemblyPolicy&&) noexcept;
        AssemblyPolicy& operator=(AssemblyPolicy&&) noexcept;

        std::vector<float> predict(const cv::Mat& robot_camera_view, const std::vector<float>& robot_joint_states);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::robotics

