#include <xinfer/compiler/drivers/hailo_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h> // Helper for temp paths

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace xinfer::compiler {

bool HailoDriver::validate_environment() {
    // Check if the Hailo SDK is installed in the current Python environment
    int res = std::system("python3 -c \"import hailo_sdk_client\" > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Python module 'hailo_sdk_client' not found.");
        XINFER_LOG_WARN("Please install the Hailo DFC: pip install hailo_sdk_client-*.whl");
        return false;
    }
    return true;
}

std::string HailoDriver::generate_python_script(const CompileConfig& config,
                                                const std::string& script_path,
                                                const std::string& arch) {
    std::ofstream py(script_path);

    py << "import sys\n";
    py << "import numpy as np\n";
    py << "from hailo_sdk_client import ClientRunner\n\n";

    py << "model_name = 'xinfer_model'\n";
    py << "onnx_path = '" << config.input_path << "'\n";
    py << "hef_path = '" << config.output_path << "'\n";
    py << "hw_arch = '" << arch << "'\n\n";

    py << "print(f'[HailoDriver] Parsing ONNX for {hw_arch}...')\n";
    py << "runner = ClientRunner(hw_arch=hw_arch)\n";
    py << "runner.translate_onnx_model(onnx_path, model_name)\n\n";

    // --- Optimization / Calibration ---
    py << "print('[HailoDriver] Starting Optimization...')\n";

    // Hailo requires a calibration function generator
    if (!config.calibration_data_path.empty()) {
        // Assume user provided a .npy file
        py << "calib_data = np.load('" << config.calibration_data_path << "')\n";
        py << "def calib_dataset():\n";
        py << "    yield [calib_data]\n\n";
        py << "runner.optimize(calib_dataset)\n";
    } else {
        // Fallback (Warning: Accuracy will be garbage)
        py << "print('WARNING: No calibration data provided! Using random data.')\n";
        py << "def random_calib():\n";
        py << "    # Creating dummy calibration data based on input shape\n";
        py << "    # Note: This is just to make the compiler pass, accuracy will be 0.\n";
        py << "    input_info = runner.get_params()['input_layers_params'][0]\n";
        py << "    shape = input_info['shape']\n";
        py << "    # Remove batch dim if present or fix it\n";
        py << "    yield [np.random.rand(1, *shape[1:]).astype(np.float32)]\n\n";
        py << "runner.optimize(random_calib)\n";
    }

    // --- Compilation ---
    py << "print('[HailoDriver] Compiling to HEF...')\n";
    py << "hef = runner.compile()\n";
    py << "with open(hef_path, 'wb') as f:\n";
    py << "    f.write(hef)\n";

    py << "print(f'[HailoDriver] HEF saved to {hef_path}')\n";

    py.close();
    return script_path;
}

bool HailoDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // 1. Determine Target Architecture
    std::string arch = "hailo8"; // Default
    for (const auto& param : config.vendor_params) {
        if (param.find("ARCH=") == 0) {
            arch = param.substr(5); // e.g. "hailo8l", "hailo15"
        }
    }

    // 2. Generate Python Script
    std::string script_path = config.output_path + ".build_hailo.py";
    generate_python_script(config, script_path, arch);

    // 3. Execute
    std::string cmd = "python3 " + script_path;
    XINFER_LOG_INFO("Running Hailo DFC via Python...");

    int result = std::system(cmd.c_str());

    // 4. Cleanup
    fs::remove(script_path);
    // Cleanup Hailo intermediate files (.har) if desired
    // fs::remove(config.output_path + ".har"); // ClientRunner creates intermediates usually

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("Hailo HEF compiled successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("Hailo compilation failed. Return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler