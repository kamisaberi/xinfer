#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Optical Flow estimation.
     */
    struct FlowResult {
        // The flow field.
        // Type: CV_32FC2 (2 channels: dx, dy)
        // Resolution: Matches input image resolution.
        cv::Mat flow_map;

        // Colorized visualization (HSV mapping)
        // Useful for debugging.
        cv::Mat visualization;
    };

    struct OpticalFlowConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., raft_small.onnx)
        std::string model_path;

        // Input Specs
        // Flow models usually need fixed dimensions divisible by 8 or 32.
        int input_width = 512;
        int input_height = 384;

        // Normalization
        // RAFT usually expects [-1, 1] range: Mean=127.5, Std=127.5
        std::vector<float> mean = {127.5f, 127.5f, 127.5f};
        std::vector<float> std  = {127.5f, 127.5f, 127.5f};
        float scale_factor = 1.0f; // Input 0-255

        // Post-processing
        bool generate_visualization = true;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class OpticalFlow {
    public:
        explicit OpticalFlow(const OpticalFlowConfig& config);
        ~OpticalFlow();

        // Move semantics
        OpticalFlow(OpticalFlow&&) noexcept;
        OpticalFlow& operator=(OpticalFlow&&) noexcept;
        OpticalFlow(const OpticalFlow&) = delete;
        OpticalFlow& operator=(const OpticalFlow&) = delete;

        /**
         * @brief Estimate flow between two frames.
         *
         * @param prev_image Frame at T-1 (BGR).
         * @param curr_image Frame at T (BGR).
         * @return FlowResult containing vectors and visualization.
         */
        FlowResult calculate(const cv::Mat& prev_image, const cv::Mat& curr_image);

        /**
         * @brief Utility: Warp an image using a flow map.
         * Moves pixels from 'src' to 'dst' based on the flow.
         */
        static cv::Mat warp(const cv::Mat& src, const cv::Mat& flow);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision