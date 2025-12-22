#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::nvidia {

/**
 * @brief NVIDIA TensorRT Backend
 * 
 * High-performance inference engine for NVIDIA GPUs.
 * 
 * Key Features:
 * - Asynchronous Execution via CUDA Streams
 * - Zero-Copy Input/Output (if using Pinned Memory)
 * - Dynamic Shape support
 * - CUDA Graph integration for low latency
 */
class NvidiaBackend : public xinfer::IBackend {
public:
    explicit NvidiaBackend(const TrtConfig& config);
    ~NvidiaBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Deserializes the .engine file and creates the Execution Context.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference asynchronously on the GPU.
     * 
     * @note This function returns immediately after launching CUDA kernels.
     * You must synchronize the stream (or wait on an event) to read results.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "NVIDIA GeForce RTX 4090")
     */
    std::string device_name() const override;

    // --- Backend Specific API ---

    /**
     * @brief Get the raw CUDA stream used by this backend.
     * Useful for synchronizing with custom CUDA kernels.
     */
    void* get_cuda_stream() const;

    /**
     * @brief Manually synchronizes the inference stream.
     * Blocks CPU until GPU finishes.
     */
    void synchronize();

private:
    // PImpl idiom to hide NvInfer/CUDA headers
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    TrtConfig m_config;
};

} // namespace xinfer::backends::nvidia