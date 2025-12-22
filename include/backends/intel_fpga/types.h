#pragma once

namespace xinfer::backends::intel_fpga {

/**
 * @brief Intel FPGA Device Family
 * Used to validate bitstream compatibility.
 */
enum class FpgaFamily {
    UNKNOWN = 0,
    ARRIA_10 = 1,     // Common in older edge cards
    STRATIX_10 = 2,   // High performance
    AGILEX = 3,       // Latest generation (High bandwidth)
    CYCLONE_V = 4     // SoC (Low power, rare for AI Suite but possible)
};

/**
 * @brief DLA Architecture Mode
 * The DLA IP can be configured for different throughput/latency trade-offs.
 */
enum class DlaArchitecture {
    GENERIC = 0,
    PE_ARRAY_STREAMING = 1, // Minimize latency (Aegis Sky)
    PE_ARRAY_BATCHED = 2    // Maximize throughput (SIEM)
};

} // namespace xinfer::backends::intel_fpga