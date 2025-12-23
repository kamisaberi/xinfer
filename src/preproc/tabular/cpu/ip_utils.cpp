#include "ip_utils.h"
#include <cstring>
#include <algorithm>

// System Headers for IPv6
#if defined(_WIN32)
    #include <winsock2.h>
    #include <ws2tcpip.h>
    #pragma comment(lib, "Ws2_32.lib")
#else
    #include <arpa/inet.h>
    #include <sys/socket.h>
#endif

namespace xinfer::preproc::tabular {

// Constant inverse for fast division
static constexpr float INV_255 = 1.0f / 255.0f;

void parse_ipv4_fast(const char* ip_str, size_t len, float* out_ptr) {
    if (!ip_str) return;

    int accumulator = 0;
    int octet_idx = 0;

    // Iterate characters
    // If len is provided, use it. If 0, check for null terminator.
    // We assume a max length of 15 ("255.255.255.255") safety cap in loop if needed.

    const char* ptr = ip_str;
    while (true) {
        char c = *ptr;

        // End of string check
        if (c == '\0' || (len > 0 && (size_t)(ptr - ip_str) >= len)) {
            // Write final octet
            out_ptr[octet_idx] = static_cast<float>(accumulator) * INV_255;
            break;
        }

        if (c >= '0' && c <= '9') {
            accumulator = accumulator * 10 + (c - '0');
        }
        else if (c == '.') {
            // Write current octet
            out_ptr[octet_idx] = static_cast<float>(accumulator) * INV_255;
            accumulator = 0;
            octet_idx++;

            if (octet_idx >= 3) {
                // Optimization: We are at the last octet.
                // Just parse the rest as number without checking for '.'
                ptr++; // Skip the dot

                // Parse last number loop
                accumulator = 0;
                while(true) {
                    char next_c = *ptr;
                    if (next_c >= '0' && next_c <= '9') {
                        accumulator = accumulator * 10 + (next_c - '0');
                        ptr++;
                    } else {
                        break;
                    }
                }
                out_ptr[octet_idx] = static_cast<float>(accumulator) * INV_255;
                return;
            }
        }
        // Skip unknown chars (whitespace trim logic could go here)

        ptr++;
    }
}

void parse_ipv6_fast(const char* ip_str, float* out_ptr) {
    if (!ip_str) return;

    // Buffer for binary bytes (IPv6 is 128-bit = 16 bytes)
    uint8_t bytes[16];

    // Initialize to zero (in case parsing fails)
    std::memset(bytes, 0, 16);

    // Use system inet_pton (Robust handling of "::", hex, etc.)
    int result = inet_pton(AF_INET6, ip_str, bytes);

    if (result == 1) {
        // Successful parse. Convert bytes to floats [0-1]
        for (int i = 0; i < 16; ++i) {
            out_ptr[i] = static_cast<float>(bytes[i]) * INV_255;
        }
    } else {
        // Fallback: If parsing failed (e.g. malformed log), fill with -1.0
        // This tells the neural net "This is garbage data"
        for (int i = 0; i < 16; ++i) {
            out_ptr[i] = -1.0f;
        }
    }
}

} // namespace xinfer::preproc::tabular