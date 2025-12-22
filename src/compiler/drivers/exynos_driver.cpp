#include <xinfer/compiler/drivers/exynos_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <cstdlib>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

// Helper to find the actual binary name since Samsung SDKs change naming often
static fs::path find_compiler_binary(const fs::path& sdk_bin_dir) {
    std::vector<std::string> candidates = {"enn_converter", "scvt", "enn_compiler"};
    for (const auto& name : candidates) {
        fs::path p = sdk_bin_dir / name;
        if (fs::exists(p)) return p;
    }
    return "";
}

bool ExynosDriver::validate_environment() {
    // 1. Check Env Var
    const char* sdk_root = std::getenv("ENN_SDK_ROOT");
    if (!sdk_root) {
        XINFER_LOG_ERROR("ENN_SDK_ROOT environment variable not set.");
        return false;
    }

    // 2. Check for Compiler Binary
    fs::path bin_dir = fs::path(sdk_root) / "bin";
    fs::path compiler = find_compiler_binary(bin_dir);

    if (compiler.empty()) {
        XINFER_LOG_ERROR("Exynos compiler not found in " + bin_dir.string());
        XINFER_LOG_WARN("Expected 'enn_converter' or 'scvt'.");
        return false;
    }

    return true;
}

bool ExynosDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input ONNX model not found: " + config.input_path);
        return false;
    }

    const char* sdk_root = std::getenv("ENN_SDK_ROOT");
    fs::path compiler_bin = find_compiler_binary(fs::path(sdk_root) / "bin");

    std::stringstream cmd;
    cmd << compiler_bin.string();

    // --- Input Model ---
    // Usage: enn_converter --input_model=model.onnx --output_model=model.nnc
    cmd << " --input_model=" << quote(config.input_path);
    cmd << " --output_model=" << quote(config.output_path);

    // --- Framework Type ---
    // Most ENN tools require specifying the source framework
    fs::path input_p(config.input_path);
    if (input_p.extension() == ".onnx") {
        cmd << " --framework=ONNX";
    } else if (input_p.extension() == ".tflite") {
        cmd << " --framework=TFLITE";
    }

    // --- Precision / Quantization ---
    // Samsung ENN usually supports "quantization aware training" or "post training quantization"
    if (config.precision == Precision::INT8) {
        cmd << " --quantize=true";
        if (!config.calibration_data_path.empty()) {
            cmd << " --data_path=" << quote(config.calibration_data_path);
        } else {
            XINFER_LOG_WARN("INT8 requested without calibration data. ENN compiler may use dummy data or fail.");
        }
    } else if (config.precision == Precision::FP16) {
        cmd << " --precision=FP16";
    }

    // --- Vendor Specific Params ---
    // Handle NPU version targeting (e.g., Exynos 2400 vs 2200)
    for (const auto& param : config.vendor_params) {
        // Example: "TARGET=exynos2400"
        if (param.find("TARGET=") == 0) {
            cmd << " --target_soc=" << param.substr(7);
        }
        // Example: "BATCH=4"
        else if (param.find("BATCH=") == 0) {
             cmd << " --batch_size=" << param.substr(6);
        }
        // Raw flags
        else {
             cmd << " " << param;
        }
    }

    // --- Execution ---
    XINFER_LOG_INFO("Executing Exynos Compiler: " + cmd.str());

    // Set LD_LIBRARY_PATH for the compiler execution
    // (Often the converter depends on libs in ../lib)
    std::string ld_preload = "export LD_LIBRARY_PATH=$ENN_SDK_ROOT/lib:$LD_LIBRARY_PATH && ";

    int result = std::system((ld_preload + cmd.str()).c_str());

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("Exynos NNC Binary generated successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("enn_converter failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler