#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h> // We use the core Tensor for output

namespace xinfer::zoo::generative {

    /**
     * @class DCGAN_Generator
     * @brief A high-level, hyper-optimized pipeline for DCGAN image generation.
     *
     * This class loads a pre-built TensorRT engine for a DCGAN generator and
     * provides a simple, one-line `generate()` function that handles noise
     * creation and inference at maximum speed.
     */
    class DCGAN_Generator {
    public:
        /**
         * @brief Constructor for the DCGAN_Generator.
         * @param engine_path Path to the pre-built, optimized TensorRT .engine file.
         */
        explicit DCGAN_Generator(const std::string& engine_path);

        ~DCGAN_Generator();

        // Rule of Five for proper resource management with PIMPL
        DCGAN_Generator(const DCGAN_Generator&) = delete;
        DCGAN_Generator& operator=(const DCGAN_Generator&) = delete;
        DCGAN_Generator(DCGAN_Generator&&) noexcept;
        DCGAN_Generator& operator=(DCGAN_Generator&&) noexcept;

        /**
         * @brief Generates a batch of images from random noise.
         * @param batch_size The number of images to generate. Must be less than or
         *                   equal to the max batch size the engine was built with.
         * @return A core::Tensor on the GPU containing the batch of generated images.
         *         The tensor will have a shape like [batch_size, 3, 64, 64].
         */
        core::Tensor generate(int batch_size = 1);

    private:
        // PIMPL (Pointer to Implementation) idiom to hide all the complex
        // CUDA and TensorRT headers from this public interface.
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::generative

