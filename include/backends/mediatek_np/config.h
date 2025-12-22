#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::mediatek {

/**
 * @brief Configuration for MediaTek NeuroPilot Backend
 */
struct MediaTekConfig {
    // Path to the compiled NeuroPilot model (.pte or .dla)
    // Must be compiled with the 'neuron_compiler' (ncc)
    std::string model_path;

    // Execution preference (Latency vs Power)
    NeuronPreference preference = NeuronPreference::PREFER_FAST_SINGLE_ANSWER;

    // Enable CPU fallback if APU ops are missing?
    bool allow_cpu_fallback = true;

    // Boost Value (0-100)
    // Manually forces higher DVFS frequencies on the APU
    int boost_value = 100;

    // Number of requests to pipeline (for batching)
    int num_inference_requests = 1;
};

} // namespace xinfer::backends::mediatek