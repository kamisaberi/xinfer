#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::exynos {

/**
 * @brief Samsung Exynos Backend
 * 
 * Executes inference on Exynos NPUs (Neural Processing Units) using the 
 * ENN (Exynos Neural Network) or Eden Runtime.
 * 
 * Supports zero-copy via ION buffers (Android) or DMABUF (Linux).
 */
class ExynosBackend : public xinfer::IBackend {
public:
    explicit ExynosBackend(const ExynosConfig& config);
    ~ExynosBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the Exynos model binary (.nnc).
     * 
     * @param model_path Path to the compiled binary.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * If inputs are standard CPU memory, copies to ION/NPU memory.
     * If inputs are MemoryType::CmaContiguous, passes handle directly.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Samsung Exynos 2400 NPU")
     */
    std::string device_name() const override;

    // --- Exynos Specific API ---

    /**
     * @brief Set the Perf/Power mode dynamically.
     * Useful for toggling boost mode when an object is detected.
     */
    void set_power_mode(EnnPowerMode mode);

private:
    // PImpl idiom to hide ENN SDK headers
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    ExynosConfig m_config;
};

} // namespace xinfer::backends::exynos