#pragma once

#include <string>
#include <cstdint>

namespace xinfer::preproc::tabular {

    /**
     * @brief Optimized IP Address Parsers for SIEM Log Processing.
     *
     * Converts IP strings into normalized float arrays [0.0, 1.0].
     * Designed to avoid memory allocations (std::string copies) during parsing.
     */

    /**
     * @brief Parses IPv4 string into 4 normalized floats.
     *
     * Example: "192.168.1.1" -> [0.753, 0.659, 0.004, 0.004]
     *
     * @param ip_str Pointer to the C-string IP address.
     * @param len Length of the string (or 0 to determine automatically).
     * @param out_ptr Pointer to a float buffer (Must hold at least 4 floats).
     */
    void parse_ipv4_fast(const char* ip_str, size_t len, float* out_ptr);

    /**
     * @brief Parses IPv6 string into 16 normalized floats.
     *
     * Handles '::' compression automatically.
     * Example: "2001:db8::1" -> [0.12, ..., 0.0, 0.004] (16 floats)
     *
     * @param ip_str Pointer to the C-string IP address.
     * @param out_ptr Pointer to a float buffer (Must hold at least 16 floats).
     */
    void parse_ipv6_fast(const char* ip_str, float* out_ptr);

} // namespace xinfer::preproc::tabular