#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::openvino {

/**
 * @brief Intel OpenVINO Backend
 * 
 * High-performance inference engine for Intel Hardware.
 * Loads Intermediate Representation (IR) format (.xml/.bin).
 * 
 * Supports:
 * - Dynamic Batching
 * - Asynchronous Inference
 * - Multi-Device Execution (CPU, GPU, NPU)
 */
class OpenVINOBackend : public xinfer::IBackend {
public:
    explicit OpenVINOBackend(const OpenVINOConfig& config);
    ~OpenVINOBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the OpenVINO IR model (.xml).
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * Wraps xInfer Tensors into ov::Tensor objects.
     * Uses zero-copy where possible (system memory sharing).
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Intel Arc A770")
     */
    std::string device_name() const override;

    /**
     * @brief Returns the layout of the input (e.g., "NCHW" or "NHWC").
     * Useful for pre-processing logic to know if transposition is needed.
     */
    std::string get_input_layout(size_t index) const;

private:
    // PImpl idiom to hide <openvino/openvino.hpp>
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    OpenVINOConfig m_config;
};

} // namespace xinfer::backends::openvino