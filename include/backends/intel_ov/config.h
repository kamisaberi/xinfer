#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::openvino {

/**
 * @brief Configuration for Intel OpenVINO Backend
 */
struct OpenVINOConfig {
    // Path to the .xml file (Topology). 
    // The .bin file (Weights) is assumed to be in the same directory.
    std::string model_path;

    // Target Hardware
    DeviceType device_type = DeviceType::CPU;

    // Optimization Hint
    PerformanceHint perf_hint = PerformanceHint::LATENCY;

    // Number of inference streams.
    // Set to >1 for high throughput on multi-core CPUs.
    int num_streams = 1;

    // Number of CPU threads to use (0 = Auto/All)
    int num_threads = 0;

    // Path to cache directory (speeds up loading on subsequent runs)
    // e.g., "./cache/ov_kernels"
    std::string cache_dir;

    // Optional: Explicitly force a specific precision (e.g., force FP16 on GPU)
    OvPrecision inference_precision = OvPrecision::FP32;
};

} // namespace xinfer::backends::openvino