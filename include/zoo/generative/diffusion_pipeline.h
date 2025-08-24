#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include <include/core/tensor.h>

namespace xinfer::zoo::generative {

    struct DiffusionPipelineConfig {
        std::string unet_engine_path;
        int num_timesteps = 50;
        int input_width = 64;
        int input_height = 64;
    };

    class DiffusionPipeline {
    public:
        explicit DiffusionPipeline(const DiffusionPipelineConfig& config);
        ~DiffusionPipeline();

        DiffusionPipeline(const DiffusionPipeline&) = delete;
        DiffusionPipeline& operator=(const DiffusionPipeline&) = delete;
        DiffusionPipeline(DiffusionPipeline&&) noexcept;
        DiffusionPipeline& operator=(DiffusionPipeline&&) noexcept;

        core::Tensor generate(int batch_size = 1);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

