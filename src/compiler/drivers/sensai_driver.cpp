#include <xinfer/compiler/drivers/sensai_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstdlib>
#include <vector>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool LatticeDriver::validate_environment() {
    // 1. Check Env Var
    const char* sdk_root = std::getenv("LATTICE_SDK_ROOT");
    if (!sdk_root) {
        XINFER_LOG_ERROR("LATTICE_SDK_ROOT environment variable not set.");
        return false;
    }

    // 2. Check for Compiler Executable
    // Name varies by version, usually 'sensai_compiler' or 'nn_compiler'
    fs::path compiler = fs::path(sdk_root) / "bin" / "sensai_compiler";
    if (!fs::exists(compiler)) {
        // Try fallback location (Windows/Linux standard install paths)
        XINFER_LOG_ERROR("Lattice compiler binary not found at: " + compiler.string());
        return false;
    }

    return true;
}

bool LatticeDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // Lattice Hardware is optimized for TFLite INT8
    fs::path in_p(config.input_path);
    if (in_p.extension() != ".tflite") {
        XINFER_LOG_WARN("Lattice sensAI prefers .tflite inputs. You provided " + in_p.extension().string());
        XINFER_LOG_WARN("Ensure your ONNX/model is fully quantized to INT8 before compilation.");
    }

    const char* sdk_root = std::getenv("LATTICE_SDK_ROOT");
    fs::path compiler_bin = fs::path(sdk_root) / "bin" / "sensai_compiler";

    std::stringstream cmd;
    cmd << compiler_bin.string();

    // --- Input / Output ---
    cmd << " --input_model " << quote(config.input_path);
    cmd << " --output_file " << quote(config.output_path);

    // --- Precision ---
    // Lattice accelerator is fixed-point.
    if (config.precision != Precision::INT8) {
        XINFER_LOG_WARN("Lattice FPGA requires INT8. Forcing compiler mode to Quantized.");
    }
    cmd << " --quantization_mode int8";

    // --- Device Family (Critical) ---
    // Users must specify target via vendor_params
    // Defaults to CrossLink-NX if not found
    bool device_set = false;
    for (const auto& param : config.vendor_params) {
        if (param.find("DEVICE=") == 0) {
            cmd << " --device " << param.substr(7); // e.g. "CrossLink-NX"
            device_set = true;
        }
        else if (param.find("IP_CONFIG=") == 0) {
            // Select IP core variant (Compact, Performance)
            cmd << " --ip_config " << param.substr(10);
        }
        else {
            cmd << " " << param;
        }
    }

    if (!device_set) {
        XINFER_LOG_WARN("No target DEVICE specified. Defaulting to 'CrossLink-NX'.");
        cmd << " --device CrossLink-NX";
    }

    // --- Execution ---
    XINFER_LOG_INFO("Executing Lattice sensAI Compiler...");
    XINFER_LOG_INFO("Cmd: " + cmd.str());

    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("Lattice Command Stream generated: " + config.output_path);
        return true;
    } else {
        XINFER_LOG_ERROR("sensai_compiler failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler