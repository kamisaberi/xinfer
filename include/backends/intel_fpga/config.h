#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::intel_fpga {

/**
 * @brief Configuration for Intel FPGA AI Suite Backend
 */
struct IntelFpgaConfig {
    // Path to the compiled model graph (.bin) produced by dla_compiler
    std::string model_path;

    // Path to the FPGA Bitstream (.aocx or .sof)
    // If empty, assumes the FPGA is already programmed.
    std::string bitstream_path;

    // Target FPGA Family
    FpgaFamily device_family = FpgaFamily::AGILEX;

    // Specific PCIe device BDF (Bus:Device.Function) if multiple cards exist.
    // e.g., "0000:03:00.0"
    std::string pcie_address;

    // Number of inferences to run in parallel on the FPGA (if hardware supports multigraph)
    int num_pipelines = 1;

    // Timeout for DLA execution in milliseconds
    uint32_t timeout_ms = 5000;
    
    // Use Shared Virtual Memory (SVM) if supported by the board (Zero-Copy)
    bool use_svm = true;
};

} // namespace xinfer::backends::intel_fpga