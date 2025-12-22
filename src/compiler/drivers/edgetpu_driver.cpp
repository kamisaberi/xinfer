#include <xinfer/compiler/drivers/edgetpu_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h> // Helper fs::path

#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool EdgeTpuDriver::validate_environment() {
    // Check if the compiler is in the PATH
    int res = std::system("edgetpu_compiler --version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Tool 'edgetpu_compiler' not found. Please install it via apt-get.");
        return false;
    }
    return true;
}

bool EdgeTpuDriver::compile(const CompileConfig& config) {
    // 1. Validation
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    fs::path in_path(config.input_path);
    if (in_path.extension() != ".tflite") {
        XINFER_LOG_ERROR("Edge TPU Compiler requires a .tflite file as input.");
        XINFER_LOG_WARN("If you have ONNX, please convert it to Quantized TFLite first using TensorFlow.");
        return false;
    }

    if (config.precision != Precision::INT8) {
        XINFER_LOG_WARN("Edge TPU requires INT8 precision. Ensure your input .tflite is already quantized.");
    }

    // 2. Prepare Temporary Output Directory
    // edgetpu_compiler forces the output filename to be <stem>_edgetpu.tflite
    // We compile to a temp dir so we can rename it later.
    fs::path temp_dir = fs::temp_directory_path() / "xinfer_edgetpu_build";
    if (fs::exists(temp_dir)) fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);

    // 3. Construct Command
    std::stringstream cmd;
    cmd << "edgetpu_compiler";
    cmd << " --out_dir " << quote(temp_dir.string());
    cmd << " --show_operations"; // Useful log output

    // Vendor Flags
    for (const auto& param : config.vendor_params) {
        if (param.find("MIN_RUNTIME=") == 0) {
            cmd << " --min_runtime_version " << param.substr(12);
        }
    }

    cmd << " " << quote(config.input_path);

    // 4. Execute
    XINFER_LOG_INFO("Running Edge TPU Compiler...");
    int result = std::system(cmd.str().c_str());

    if (result != 0) {
        XINFER_LOG_ERROR("edgetpu_compiler failed. Check if input is fully quantized.");
        return false;
    }

    // 5. Locate and Move Output
    // Expected filename: <input_stem>_edgetpu.tflite
    std::string expected_name = in_path.stem().string() + "_edgetpu.tflite";
    fs::path generated_file = temp_dir / expected_name;

    if (!fs::exists(generated_file)) {
        XINFER_LOG_ERROR("Compiler finished but output file missing: " + generated_file.string());
        return false;
    }

    // Move to user's requested output path
    try {
        fs::path final_out(config.output_path);

        // Ensure parent dir exists
        if (final_out.has_parent_path()) {
            fs::create_directories(final_out.parent_path());
        }

        fs::rename(generated_file, final_out);
        fs::remove_all(temp_dir); // Cleanup

        XINFER_LOG_SUCCESS("Edge TPU Model compiled: " + config.output_path);
        return true;

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Failed to move output file: " + std::string(e.what()));
        return false;
    }
}

} // namespace xinfer::compiler