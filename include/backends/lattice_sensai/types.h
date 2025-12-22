#pragma once

namespace xinfer::backends::lattice {

/**
 * @brief Lattice FPGA Device Family
 * Determines the instruction set compatibility for the Neural Network Accelerator.
 */
enum class DeviceFamily {
    UNKNOWN = 0,
    ICE40_ULTRAPLUS = 1, // Ultra-low power (mW range)
    ECP5 = 2,            // General purpose, higher capacity
    CROSSLINK_NX = 3,    // FD-SOI, High perf/watt
    CERTUS_NX = 4
};

/**
 * @brief Host-to-FPGA Communication Interface
 * How xInfer sends input tensors to the FPGA.
 */
enum class ConnectionInterface {
    USB_FTDI = 0,    // Standard dev boards (via libftdi/libusb)
    SPI_DEV = 1,     // Embedded Linux host (e.g., Raspberry Pi -> SPI -> Lattice)
    I2C_DEV = 2,     // Slow, control messages only
    PCIE = 3         // ECP5/Certus-NX PCIe cards
};

} // namespace xinfer::backends::lattice