#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::microchip {

/**
 * @brief Microchip VectorBlox Backend
 * 
 * Executes inference on PolarFire FPGAs using the VectorBlox CNN IP.
 * 
 * Features:
 * - Deterministic Latency (No OS jitter on the accelerator)
 * - Support for INT8 execution
 * - "Soft" overlay allows running different models without reprogramming the bitstream
 */
class VectorBloxBackend : public xinfer::IBackend {
public:
    explicit VectorBloxBackend(const VectorBloxConfig& config);
    ~VectorBloxBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Initializes the VBX Core and loads the model blob into DDR.
     * 
     * @param model_path Path to the binary blob.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * Flushes CPU caches (if on PolarFire SoC) and triggers the V1000/V2000 engine.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Microchip VectorBlox V1000")
     */
    std::string device_name() const override;

    /**
     * @brief Get the internal temperature of the FPGA die (if supported by system monitor).
     */
    float get_fpga_temp() const;

private:
    // PImpl idiom to hide VectorBlox SDK headers
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    VectorBloxConfig m_config;
};

} // namespace xinfer::backends::microchip