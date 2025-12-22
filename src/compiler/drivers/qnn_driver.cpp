#include <xinfer/compiler/drivers/qnn_driver.h>
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

std::string QnnDriver::get_tool_path(const std::string& tool_name) {
    const char* sdk_root = std::getenv("QNN_SDK_ROOT");
    if (!sdk_root) return "";

    // QNN tools are usually in $QNN_SDK_ROOT/bin/x86_64-linux-clang/
    fs::path bin_dir = fs::path(sdk_root) / "bin" / "x86_64-linux-clang";
    return (bin_dir / tool_name).string();
}

bool QnnDriver::validate_environment() {
    // 1. Check QNN SDK
    const char* sdk_root = std::getenv("QNN_SDK_ROOT");
    if (!sdk_root) {
        XINFER_LOG_ERROR("QNN_SDK_ROOT environment variable not set.");
        return false;
    }

    // 2. Check Android NDK (Required for qnn-model-lib-generator)
    // The lib generator uses clang from the NDK to build the model shared object
    const char* ndk_root = std::getenv("ANDROID_NDK_ROOT");
    if (!ndk_root) {
        // Fallback check for standard NDK locations?
        XINFER_LOG_WARN("ANDROID_NDK_ROOT not set. 'qnn-model-lib-generator' requires the NDK to compile graph sources.");
    }

    // 3. Check for specific binary
    std::string converter = get_tool_path("qnn-onnx-converter");
    if (!fs::exists(converter)) {
        XINFER_LOG_ERROR("QNN tool not found at: " + converter);
        return false;
    }

    return true;
}

bool QnnDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // Setup Temporary Directory
    fs::path build_dir = fs::temp_directory_path() / "xinfer_qnn_build";
    if (fs::exists(build_dir)) fs::remove_all(build_dir);
    fs::create_directories(build_dir);

    std::string model_name = "qnn_model";

    // --- Step 1: Convert ONNX to C++ Graph Source ---
    {
        XINFER_LOG_INFO("[Step 1/3] Running qnn-onnx-converter...");
        std::stringstream cmd;
        cmd << get_tool_path("qnn-onnx-converter");
        cmd << " --input_network " << quote(config.input_path);
        cmd << " --output_path " << quote((build_dir / (model_name + ".cpp")).string());

        // Handling INT8 Quantization (requires input_list)
        if (config.precision == Precision::INT8) {
            if (!config.calibration_data_path.empty()) {
                cmd << " --input_list " << quote(config.calibration_data_path);
                // Usually requires --quantization_overrides or params for quantization scheme
                // Defaulting to standard scheme
                cmd << " --param_quantizer_type enhanced";
            } else {
                XINFER_LOG_WARN("INT8 requested but no calibration list provided to qnn-onnx-converter.");
            }
        }

        if (std::system(cmd.str().c_str()) != 0) {
            XINFER_LOG_ERROR("qnn-onnx-converter failed.");
            return false;
        }
    }

    // --- Step 2: Compile C++ Source to Model Library (.so) ---
    {
        XINFER_LOG_INFO("[Step 2/3] Running qnn-model-lib-generator...");
        std::stringstream cmd;
        cmd << get_tool_path("qnn-model-lib-generator");
        cmd << " -c " << quote((build_dir / (model_name + ".cpp")).string());
        cmd << " -b " << quote((build_dir / (model_name + ".bin")).string()); // weights
        cmd << " -o " << quote(build_dir.string());
        cmd << " -l " << model_name;

        // QNN generator needs 'libQnnHtp.so' or 'libQnnCpu.so' specified if generating specific binaries
        // but here we just generate the generic model lib.

        if (std::system(cmd.str().c_str()) != 0) {
            XINFER_LOG_ERROR("qnn-model-lib-generator failed. Check ANDROID_NDK_ROOT.");
            return false;
        }
    }

    // --- Step 3: Generate Context Binary (Serialized Engine) ---
    {
        XINFER_LOG_INFO("[Step 3/3] Running qnn-context-binary-generator...");

        std::string backend_lib = "libQnnHtp.so"; // Default to HTP (NPU)
        for (const auto& param : config.vendor_params) {
            if (param == "BACKEND=CPU") backend_lib = "libQnnCpu.so";
            if (param == "BACKEND=GPU") backend_lib = "libQnnGpu.so";
        }

        std::stringstream cmd;
        cmd << get_tool_path("qnn-context-binary-generator");

        // Input is the .so we just built in Step 2
        // Note: The generator usually outputs to <dir>/<target_arch>/lib<name>.so
        // We assume aarch64-android target for QNN usually.
        fs::path lib_path = build_dir / "aarch64-android" / ("lib" + model_name + ".so");
        if (!fs::exists(lib_path)) {
             // Try finding it if arch name differs
             // Logic to find generated .so omitted for brevity
             XINFER_LOG_ERROR("Could not find generated lib: " + lib_path.string());
             return false;
        }

        cmd << " --model " << quote(lib_path.string());
        cmd << " --backend " << backend_lib; // The backend to compile FOR
        cmd << " --output_dir " << quote(build_dir.string());
        cmd << " --binary_file " << "final_context";

        if (std::system(cmd.str().c_str()) != 0) {
            XINFER_LOG_ERROR("qnn-context-binary-generator failed.");
            return false;
        }
    }

    // --- Step 4: Finalize ---
    fs::path context_bin = build_dir / "final_context.bin";
    if (fs::exists(context_bin)) {
        try {
            fs::copy_file(context_bin, config.output_path, fs::copy_options::overwrite_existing);
            fs::remove_all(build_dir); // Cleanup
            XINFER_LOG_SUCCESS("QNN Context Binary compiled: " + config.output_path);
            return true;
        } catch (const std::exception& e) {
            XINFER_LOG_ERROR("Failed to move output file: " + std::string(e.what()));
            return false;
        }
    } else {
        XINFER_LOG_ERROR("Context binary generation failed to produce output.");
        return false;
    }
}

} // namespace xinfer::compiler