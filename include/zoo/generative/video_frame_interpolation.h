#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct InterpolationConfig {
        // Hardware Target (GPU is required for real-time performance)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., rife_v4.engine)
        // Model takes two frames (and optical flow) as input.
        std::string model_path;

        // Input Specs (Model dependent)
        int input_width = 448;
        int input_height = 256;

        // Upscaling Factor
        // 2x = 30->60fps, 4x = 30->120fps
        int upscale_factor = 2;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class FrameInterpolator {
    public:
        explicit FrameInterpolator(const InterpolationConfig& config);
        ~FrameInterpolator();

        // Move semantics
        FrameInterpolator(FrameInterpolator&&) noexcept;
        FrameInterpolator& operator=(FrameInterpolator&&) noexcept;
        FrameInterpolator(const FrameInterpolator&) = delete;
        FrameInterpolator& operator=(const FrameInterpolator&) = delete;

        /**
         * @brief Generate intermediate frames between two input frames.
         *
         * @param frame0 The starting frame (t=0).
         * @param frame1 The ending frame (t=1).
         * @return A vector containing the newly synthesized frames.
         *         If upscale_factor=2, returns one frame at t=0.5.
         *         If upscale_factor=4, returns three frames at t=0.25, 0.5, 0.75.
         */
        std::vector<cv::Mat> interpolate(const cv::Mat& frame0, const cv::Mat& frame1);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative