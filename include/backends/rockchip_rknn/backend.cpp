#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::rknn {

/**
 * @brief Rockchip RKNN Backend
 * 
 * High-performance inference for RK35xx Series.
 * 
 * Features:
 * - Multi-Core NPU scheduling (RK3588)
 * - Zero-Copy support via DRM buffers (critical for 4K video analysis)
 * - INT8 Quantized execution
 */
class RknnBackend : public xinfer::IBackend {
public:
    explicit RknnBackend(const RknnConfig& config);
    ~RknnBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the .rknn model.
     * 
     * @param model_path Path to the .rknn file.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * If configured for Zero-Copy and inputs are CmaContiguous,
     * it passes physical addresses directly to the NPU.
     * Otherwise, performs a memcpy to internal NPU buffers.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Rockchip RK3588 NPU")
     */
    std::string device_name() const override;

    // --- Backend Specific API ---

    /**
     * @brief Get SDK Version (e.g., "1.5.0")
     */
    std::string get_sdk_version() const;

private:
    // PImpl idiom to hide rknn_api.h
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    RknnConfig m_config;
};

} // namespace xinfer::backends::rknn