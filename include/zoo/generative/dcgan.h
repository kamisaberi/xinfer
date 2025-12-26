#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::generative {

    struct DcganConfig {
        // Hardware Target (GANs are heavy, GPU preferred)
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., dcgan_generator.engine)
        // Expected Input: [Batch, LatentDim] (e.g., [1, 100])
        // Expected Output: [Batch, Channels, H, W]
        std::string model_path;

        // Input Specs
        int latent_dim = 100; // Size of the input noise vector

        // Generation settings
        int seed = -1; // -1 for random seed, or fixed for reproducible results

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class DCGAN {
    public:
        explicit DCGAN(const DcganConfig& config);
        ~DCGAN();

        // Move semantics
        DCGAN(DCGAN&&) noexcept;
        DCGAN& operator=(DCGAN&&) noexcept;
        DCGAN(const DCGAN&) = delete;
        DCGAN& operator=(const DCGAN&) = delete;

        /**
         * @brief Generate a new image from a random seed.
         *
         * @return Synthesized image (BGR, uint8).
         */
        cv::Mat generate();

        /**
         * @brief Generate an image from a specific latent vector.
         * Useful for latent space interpolation.
         */
        cv::Mat generate_from_vector(const std::vector<float>& latent_vector);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative