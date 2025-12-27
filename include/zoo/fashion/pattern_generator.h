#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::fashion {

    struct PatternConfig {
        // Hardware Target (GANs run best on GPU)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., stylegan2_patterns.engine)
        std::string model_path;

        // Input Specs
        int latent_dim = 512; // Standard for StyleGAN

        // Output Specs
        int output_width = 1024;
        int output_height = 1024;

        // Generation settings
        int seed = -1; // -1 for random, fixed for reproducibility

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PatternGenerator {
    public:
        explicit PatternGenerator(const PatternConfig& config);
        ~PatternGenerator();

        // Move semantics
        PatternGenerator(PatternGenerator&&) noexcept;
        PatternGenerator& operator=(PatternGenerator&&) noexcept;
        PatternGenerator(const PatternGenerator&) = delete;
        PatternGenerator& operator=(const PatternGenerator&) = delete;

        /**
         * @brief Generate a new, random pattern.
         *
         * @return Synthesized pattern image (BGR).
         */
        cv::Mat generate();

        /**
         * @brief Generate a pattern from a specific seed vector.
         * Allows for interpolation and style mixing.
         *
         * @param latent_vector The input noise vector.
         * @return Synthesized pattern image.
         */
        cv::Mat generate_from_vector(const std::vector<float>& latent_vector);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::fashion