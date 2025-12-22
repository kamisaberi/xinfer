#include <xinfer/compiler/drivers/coreml_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h> // For file_exists, temp_directory

#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool CoreMLDriver::validate_environment() {
    // 1. Check for macOS Compiler (mlmodel -> mlmodelc)
    // Only available on macOS. On Linux, this driver acts purely as a transpiler if needed.
#ifdef __APPLE__
    int res = std::system("xcrun coremlcompiler version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("'xcrun coremlcompiler' not found. Install Xcode Command Line Tools.");
        return false;
    }
#else
    XINFER_LOG_WARN("Not running on macOS. Cannot compile final .mlmodelc bundle, only conversion to .mlmodel is supported via Python.");
#endif

    // 2. Check for Python coremltools (ONNX -> mlmodel)
    int py_res = std::system("python3 -c \"import coremltools\" > /dev/null 2>&1");
    if (py_res != 0) {
        XINFER_LOG_WARN("Python package 'coremltools' not found. ONNX conversion will fail.");
        // We don't return false here because user might provide a pre-made .mlmodel file
    }

    return true;
}

bool CoreMLDriver::convert_onnx_to_mlmodel(const std::string& onnx_path,
                                           const std::string& mlmodel_path,
                                           Precision precision) {
    XINFER_LOG_INFO("Converting ONNX to Core ML (.mlmodel)...");

    // Generate a temporary Python script
    std::string script_path = "/tmp/xinfer_convert_coreml.py"; // Use proper temp dir in prod
    std::ofstream script(script_path);

    script << "import coremltools as ct\n";
    script << "import onnx\n";
    script << "import sys\n";

    script << "try:\n";
    script << "    model_onnx = onnx.load('" << onnx_path << "')\n";

    // Convert
    script << "    # Precision handling\n";
    if (precision == Precision::FP16) {
        script << "    model = ct.converters.onnx.convert(model_onnx, minimum_ios_deployment_target='13.0')\n";
        script << "    # Note: Modern coremltools handles FP16 automatically for neural engine\n";
    } else {
        script << "    model = ct.converters.onnx.convert(model_onnx)\n";
    }

    script << "    model.save('" << mlmodel_path << "')\n";
    script << "    print('Conversion successful')\n";
    script << "except Exception as e:\n";
    script << "    print(f'Error: {e}')\n";
    script << "    sys.exit(1)\n";

    script.close();

    // Run Script
    std::string cmd = "python3 " + script_path;
    int res = std::system(cmd.c_str());

    // Cleanup
    std::remove(script_path.c_str());

    return (res == 0 && fs::exists(mlmodel_path));
}

bool CoreMLDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    fs::path input_path(config.input_path);
    std::string intermediate_mlmodel = config.input_path;
    bool temp_file_created = false;

    // Step 1: If input is ONNX, convert to .mlmodel first
    if (input_path.extension() == ".onnx") {
        intermediate_mlmodel = config.output_path + ".temp.mlmodel"; // Temporary intermediate
        if (!convert_onnx_to_mlmodel(config.input_path, intermediate_mlmodel, config.precision)) {
            XINFER_LOG_ERROR("Failed to convert ONNX to .mlmodel");
            return false;
        }
        temp_file_created = true;
    }

    // Step 2: Compile .mlmodel -> .mlmodelc
#ifdef __APPLE__
    XINFER_LOG_INFO("Compiling .mlmodel to .mlmodelc bundle...");

    std::stringstream cmd;
    cmd << "xcrun coremlcompiler compile ";
    cmd << quote(intermediate_mlmodel) << " ";

    // Output folder
    fs::path out_path(config.output_path);
    cmd << quote(out_path.parent_path().string());

    int res = std::system(cmd.str().c_str());

    // Cleanup intermediate
    if (temp_file_created) {
        fs::remove(intermediate_mlmodel);
    }

    if (res == 0) {
        // xcrun outputs to <Destination>/<ModelName>.mlmodelc
        // We might need to rename it if the user requested a specific output filename
        // logic omitted for brevity, assuming standard behavior
        XINFER_LOG_SUCCESS("Core ML compilation complete.");
        return true;
    } else {
        XINFER_LOG_ERROR("xcrun compilation failed.");
        return false;
    }

#else
    if (temp_file_created) {
         XINFER_LOG_SUCCESS("Converted to .mlmodel (Linux mode). Final compilation to .mlmodelc requires macOS.");
         XINFER_LOG_INFO("File saved to: " + intermediate_mlmodel);
         return true;
    }
    return false;
#endif
}

} // namespace xinfer::compiler