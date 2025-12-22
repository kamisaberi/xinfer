#pragma once

#include <string>
#include <vector>
#include <memory>
#include <xinfer/core/backend_interface.h>
#include <xinfer/core/tensor.h>
#include "config.h"

namespace xinfer::backends::intel_fpga {

/**
 * @brief Intel FPGA AI Suite Backend
 * 
 * Executes inference on Intel FPGAs (Agilex, Stratix 10, Arria 10) instantiated
 * with the DLA (Deep Learning Accelerator) IP Core.
 * 
 * Communication happens via the Intel FPGA Runtime (PCIe/Avalon-MM).
 */
class IntelFpgaBackend : public xinfer::IBackend {
public:
    explicit IntelFpgaBackend(const IntelFpgaConfig& config);
    ~IntelFpgaBackend() override;

    // --- Implementation of IBackend ---

    /**
     * @brief Loads the DLA binary (.bin) and optionally programs the Bitstream.
     * 
     * @param model_path Path to the .bin file from dla_compiler.
     */
    bool load_model(const std::string& model_path) override;

    /**
     * @brief Executes inference on the DLA IP.
     * 
     * Handles data transfer to the FPGA DDR or SVM (Shared Virtual Memory).
     */
    void predict(const std::vector<core::Tensor>& inputs, 
                 std::vector<core::Tensor>& outputs) override;

    /**
     * @brief Returns device name (e.g., "Intel Agilex DLA")
     */
    std::string device_name() const override;

    /**
     * @brief Reprogram the FPGA Device.
     * Useful if switching between different Aegis Sky tracking modes that require
     * different hardware architectures.
     */
    bool program_bitstream(const std::string& aocx_path);

private:
    // PImpl idiom to hide Intel FPGA Runtime headers
    struct Impl;
    std::unique_ptr<Impl> m_impl;

    IntelFpgaConfig m_config;
};

} // namespace xinfer::backends::intel_fpga