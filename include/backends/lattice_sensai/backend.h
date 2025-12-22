#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::lattice {

/**
 * @brief Lattice sensAI Backend
 * 
 * Executes inference on Lattice FPGAs running the sensAI CNN Accelerator IP.
 * 
 * Since Lattice devices are often "bare metal" without an OS, this backend
 * acts as a Host Controller:
 * 1. It uploads the NN commands (weights/instructions) to the FPGA.
 * 2. It writes input tensor data to the FPGA via USB/SPI.
 * 3. It triggers execution via a CSR (Control Status Register) write.
 * 4. It reads back results from the FPGA memory.
 */
class LatticeBackend : public xinfer::IBackend {
public:
    explicit LatticeBackend(const LatticeConfig& config);
    ~LatticeBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Connects to the FPGA and uploads the NN command stream.
     * 
     * @param model_path Path to the .bin command stream file.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference.
     * 
     * Performs a transaction: Write Input -> Trigger -> Poll Status -> Read Output.
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Lattice CrossLink-NX (USB)")
     */
    std::string device_name() const override;

    /**
     * @brief Programs the FPGA SRAM with a bitstream.
     * Useful for reconfiguration on the fly.
     */
    bool flash_bitstream(const std::string& bit_path);

private:
    // PImpl idiom to hide libftdi/spidev details
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    LatticeConfig m_config;
};

} // namespace xinfer::backends::lattice