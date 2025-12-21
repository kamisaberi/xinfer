#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include "types.h"

namespace xinfer::backends::ryzen_ai {

    /**
     * @brief Configuration for AMD Ryzen AI (XDNA) Backend
     */
    struct RyzenAIConfig {
        // Path to the compiled model (.xmodel for Native, .onnx for Vitis EP)
        std::string model_path;

        // Path to the Vitis AI Config file (usually vaip_config.json)
        // Required if using the Vitis AI Execution Provider.
        std::string config_file_path;

        // The runtime strategy to use
        RuntimeType runtime_type = RuntimeType::VITIS_AI_EP;

        // NPU Performance Profile
        XdnaProfile profile = XdnaProfile::DEFAULT;

        // Number of threads to use for CPU pre/post processing fallbacks
        int cpu_threads = 2;

        // If using Native XRT: the directory containing the AIE binaries (.xclbin)
        std::string xclbin_path;
    };

} // namespace xinfer::backends::ryzen_ai