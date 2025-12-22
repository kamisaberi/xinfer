#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::hailo {

/**
 * @brief HailoRT Backend
 * 
 * Executes inference on Hailo-8, Hailo-8L, and Hailo-10 processors.
 * Loads .hef (Hailo Executable Format) files.
 * 
 * Utilizes "VStreams" (Virtual Streams) for asynchronous data transfer
 * over PCIe/USB.
 */
class HailoBackend : public xinfer::IBackend {
public:
    explicit HailoBackend(const HailoConfig& config);
    ~HailoBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the .hef file and configures VStreams.
     * 
     * @param model_path Path to the .hef file.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * Writes data to Input VStreams and reads from Output VStreams.
     * Thread-safe if configured with the Multiplexer.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Hailo-8 PCIe")
     */
    std::string device_name() const override;

    /**
     * @brief Get the chip temperature (Diagnostic).
     */
    float get_chip_temperature() const;

private:
    // PImpl idiom to hide <hailo/hailort.hpp>
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    HailoConfig m_config;
};

} // namespace xinfer::backends::hailo