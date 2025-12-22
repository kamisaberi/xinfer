#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::lattice {

/**
 * @brief Configuration for Lattice sensAI Backend
 */
struct LatticeConfig {
    // Path to the Neural Network Command Stream (.bin)
    // This file contains the instructions for the CNN Accelerator IP.
    std::string model_path;

    // Optional: Path to the FPGA Bitstream (.bit)
    // If provided, xInfer attempts to program the FPGA before loading the model.
    std::string bitstream_path;

    // Target Hardware
    DeviceFamily family = DeviceFamily::CROSSLINK_NX;
    ConnectionInterface interface = ConnectionInterface::USB_FTDI;

    // Interface Specific Settings
    // For USB: Product ID / Serial Number
    // For SPI: Device Path (e.g., "/dev/spidev0.0")
    std::string device_address; 
    
    // Clock speed for SPI/I2C communication in Hz
    uint32_t baud_rate = 1000000; // 1MHz default

    // Address in FPGA SRAM where input data should be written
    uint32_t input_base_addr = 0x0000;
    
    // Address in FPGA SRAM where output data appears
    uint32_t output_base_addr = 0x1000;
};

} // namespace xinfer::backends::lattice