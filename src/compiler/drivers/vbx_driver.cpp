#include <xinfer/compiler/drivers/vbx_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <sstream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool VectorBloxDriver::validate_environment() {
    // 1. Check Env Var
    const char* sdk_root = std::getenv("VECTORBLOX_SDK");
    if (!sdk_root) {
        XINFER_LOG_ERROR("VECTORBLOX_SDK environment variable not set.");
        return false;
    }

    // 2. Check for Python module availability
    // The SDK typically relies on the 'vbx' python package
    int res = std::system("python3 -c \"import vbx\" > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_WARN("Python module 'vbx' not found. Ensure you have installed the SDK python requirements.");
    }

    return true;
}

bool VectorBloxDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // VectorBlox requires INT8 quantization calibration
    if (config.precision != Precision::INT8) {
        XINFER_LOG_WARN("VectorBlox IP is optimized for INT8. Using FP16/FP32 will result in extremely slow CPU fallback or failure.");
    }

    const char* sdk_root = std::getenv("VECTORBLOX_SDK");

    // We construct a python command to run the compilation script provided by the SDK.
    // Usually located at $VECTORBLOX_SDK/example/python/generate_blob.py or similar.
    // For this driver, we assume a standard script name 'vbx_compile' is reachable or we call python directly.

    std::stringstream cmd;
    cmd << "python3 -m vbx.generate_blob"; // Standard entry point for modern SDKs

    // --- Input / Output ---
    cmd << " --model " << quote(config.input_path);
    cmd << " --output " << quote(config.output_path);

    // --- Precision / Calibration ---
    // If not already quantized, VBX needs a sample dataset for Post Training Quantization (PTQ)
    if (!config.calibration_data_path.empty()) {
        cmd << " --calib_images " << quote(config.calibration_data_path);
    } else if (config.precision == Precision::INT8) {
        XINFER_LOG_WARN("INT8 requested but no calibration data provided. VBX Compiler may fail.");
    }

    // --- Vendor Specific Params (Core Configuration) ---
    // The compiler MUST know the target core size (V250, V1000, etc.)
    bool core_set = false;
    for (const auto& param : config.vendor_params) {
        // Example: "CORE=V1000"
        if (param.find("CORE=") == 0) {
            cmd << " --core " << param.substr(5);
            core_set = true;
        }
        // Example: "HEIGHT=480" (Fixed resolution required for FPGA)
        else if (param.find("HEIGHT=") == 0) {
             cmd << " --height " << param.substr(7);
        }
        else if (param.find("WIDTH=") == 0) {
             cmd << " --width " << param.substr(6);
        }
        else {
             cmd << " " << param;
        }
    }

    if (!core_set) {
        XINFER_LOG_WARN("No target CORE specified (e.g., CORE=V1000). Defaulting to V1000.");
        cmd << " --core V1000";
    }

    // --- Execution ---
    XINFER_LOG_INFO("Executing VectorBlox Compiler...");
    XINFER_LOG_INFO("Cmd: " + cmd.str());

    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("VectorBlox BLOB generated successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("VectorBlox compilation failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler