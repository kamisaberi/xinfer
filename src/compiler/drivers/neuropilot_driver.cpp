#include <xinfer/compiler/drivers/neuropilot_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool NeuroPilotDriver::validate_environment() {
    // Check if 'ncc' is in the PATH
    // The NeuroPilot SDK adds this tool to the path when sourced.
    int res = std::system("ncc --version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Tool 'ncc' (Neuron Compiler) not found.");
        XINFER_LOG_WARN("Please install MediaTek NeuroPilot SDK and source the env script.");
        return false;
    }
    return true;
}

bool NeuroPilotDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    std::stringstream cmd;
    cmd << "ncc compile";

    // --- Input / Output ---
    cmd << " " << quote(config.input_path);
    cmd << " -o " << quote(config.output_path);

    // --- Optimization Level ---
    // -O1: Fast compile, -O2: Better performance (default), -O3: Max optimization
    cmd << " -O2";

    // --- Precision ---
    // NeuroPilot heavily favors INT8.
    // If FP16 is requested, we can use --relax-fp32 to allow lower precision fallback.
    if (config.precision == Precision::FP16) {
        cmd << " --relax-fp32";
    }

    // --- Architecture (Crucial for MediaTek) ---
    // Users must specify which APU generation they are targeting via vendor_params.
    // e.g., ARCH=mdla3.0 (Genio 1200)
    bool arch_set = false;
    for (const auto& param : config.vendor_params) {
        if (param.find("ARCH=") == 0) {
            cmd << " --arch=" << param.substr(5);
            arch_set = true;
        }
        else if (param.find("platform=") == 0) {
             // Legacy way to specify platform
             cmd << " --platform " << param.substr(9);
        }
        else {
             // Pass raw flags (e.g. --enable-dla-buffer)
             cmd << " " << param;
        }
    }

    if (!arch_set) {
        XINFER_LOG_WARN("No target architecture specified (e.g., ARCH=mdla3.0). ncc will target the host's default, which may fail if cross-compiling.");
    }

    // --- Execution ---
    XINFER_LOG_INFO("Executing NeuroPilot Compiler: " + cmd.str());

    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("NeuroPilot Binary compiled successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("ncc failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler