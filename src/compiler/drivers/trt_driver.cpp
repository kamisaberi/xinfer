#include <xinfer/compiler/drivers/trt_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h> // Assuming helpers like file_exists

#include <iostream>
#include <sstream>
#include <filesystem>
#include <vector>
#include <cstdlib> // for std::system

namespace fs = std::filesystem;

namespace xinfer::compiler {

// Helper to safely quote paths for the shell
static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool TrtDriver::validate_environment() {
    // Check if 'trtexec' is in the system PATH
    int result = std::system("trtexec --version > /dev/null 2>&1");
    if (result != 0) {
        XINFER_LOG_ERROR("Tool 'trtexec' not found. Please install NVIDIA TensorRT.");
        return false;
    }
    return true;
}

bool TrtDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input model not found: " + config.input_path);
        return false;
    }

    // 1. Build the Command Line
    std::stringstream cmd;
    cmd << "trtexec";
    
    // Input/Output
    cmd << " --onnx=" << quote(config.input_path);
    cmd << " --saveEngine=" << quote(config.output_path);

    // 2. Precision Flags
    switch (config.precision) {
        case Precision::FP16:
            cmd << " --fp16";
            break;
        case Precision::INT8:
            cmd << " --int8";
            // If a calibration cache exists, use it. 
            // Otherwise trtexec will try to run calibration (requires data loader, which CLI can't provide easily)
            if (!config.calibration_data_path.empty()) {
                cmd << " --calib=" << quote(config.calibration_data_path);
            } else {
                XINFER_LOG_WARN("INT8 requested but no calibration cache provided. trtexec may use default ranges.");
            }
            break;
        case Precision::FP32:
        default:
            // Default behavior is FP32
            break;
    }

    // 3. Optimization & System Flags
    // Standard flags for best performance in xInfer
    cmd << " --verbose"; // To see output in console
    cmd << " --noDataTransfer"; // Benchmark/Compile mode doesn't need real data transfer
    
    // 4. Vendor Specific Params (Overrides)
    // Allows user to pass "WORKSPACE=2048" or "BATCH=8" via CLI
    for (const auto& param : config.vendor_params) {
        // Example param: "WORKSPACE=1024"
        if (param.find("WORKSPACE=") == 0) {
            std::string size = param.substr(10);
            cmd << " --memPoolSize=workspace:" << size; 
        }
        else if (param.find("BATCH=") == 0) {
            // Force explicit batch size for optimization profile
            std::string batch = param.substr(6);
            cmd << " --minShapes=input:1x3x224x224"; // Example defaults, real implementation parses ONNX input name
            cmd << " --optShapes=input:" << batch << "x3x224x224";
            cmd << " --maxShapes=input:" << batch << "x3x224x224";
        }
        // Add more flags as needed (e.g., --streams, --profiling)
    }

    // 5. Execute
    XINFER_LOG_INFO("Executing: " + cmd.str());
    
    int result = std::system(cmd.str().c_str());

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("TensorRT Engine compiled successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("trtexec failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler