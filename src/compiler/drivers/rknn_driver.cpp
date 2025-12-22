#include <xinfer/compiler/drivers/rknn_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h> // Helper for temp paths

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace xinfer::compiler {

bool RknnDriver::validate_environment() {
    // Check if rknn-toolkit2 is installed in the python environment
    // We attempt to import it.
    int res = std::system("python3 -c \"from rknn.api import RKNN\" > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Python module 'rknn.api' not found.");
        XINFER_LOG_WARN("Please install rknn-toolkit2: pip install rknn-toolkit2");
        return false;
    }
    return true;
}

std::string RknnDriver::generate_python_script(const CompileConfig& config,
                                               const std::string& script_path,
                                               const std::string& target_platform) {
    std::ofstream py(script_path);

    py << "import sys\n";
    py << "from rknn.api import RKNN\n\n";

    py << "ONNX_MODEL = '" << config.input_path << "'\n";
    py << "RKNN_MODEL = '" << config.output_path << "'\n";
    py << "TARGET_PLATFORM = '" << target_platform << "'\n\n";

    py << "if __name__ == '__main__':\n";
    py << "    # 1. Create RKNN object\n";
    py << "    rknn = RKNN(verbose=True)\n\n";

    py << "    # 2. Config\n";
    py << "    print(f'--> Config for {TARGET_PLATFORM}')\n";
    // Optimization level 3 is standard for RK3588
    py << "    rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]], target_platform=TARGET_PLATFORM)\n\n";

    py << "    # 3. Load ONNX\n";
    py << "    print('--> Loading model')\n";
    // We assume input names are auto-detected.
    // If strict naming is needed, vendor_params parsing logic would go here.
    py << "    ret = rknn.load_onnx(model=ONNX_MODEL)\n";
    py << "    if ret != 0:\n";
    py << "        print('Load model failed!')\n";
    py << "        sys.exit(ret)\n\n";

    py << "    # 4. Build\n";
    py << "    print('--> Building')\n";

    // Handle Precision / Quantization
    if (config.precision == Precision::INT8) {
        py << "    # INT8 Quantization Requested\n";
        if (!config.calibration_data_path.empty()) {
            // dataset parameter expects a .txt file with image paths
            py << "    ret = rknn.build(do_quantization=True, dataset='" << config.calibration_data_path << "')\n";
        } else {
            // No calibration data? rknn-toolkit2 cannot do INT8. Fallback or Fail.
            // We force hybrid quantization or fail. Let's warn and try generic.
            py << "    print('WARNING: INT8 requested but no calibration dataset provided!')\n";
            py << "    print('Falling back to FP16 to avoid failure.')\n";
            py << "    ret = rknn.build(do_quantization=False)\n";
        }
    } else {
        // FP16 (Standard for RKNN NPU)
        py << "    ret = rknn.build(do_quantization=False)\n";
    }

    py << "    if ret != 0:\n";
    py << "        print('Build model failed!')\n";
    py << "        sys.exit(ret)\n\n";

    py << "    # 5. Export\n";
    py << "    print('--> Exporting RKNN model')\n";
    py << "    ret = rknn.export_rknn(RKNN_MODEL)\n";
    py << "    if ret != 0:\n";
    py << "        print('Export rknn failed!')\n";
    py << "        sys.exit(ret)\n\n";

    py << "    print('done')\n";

    py.close();
    return script_path;
}

bool RknnDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // 1. Determine Target Platform
    // Default to rk3588 as it's the most popular high-end chip
    std::string platform = "rk3588";

    for (const auto& param : config.vendor_params) {
        if (param.find("PLATFORM=") == 0) {
            platform = param.substr(9); // e.g. "rk3568"
        }
    }

    // 2. Generate Python Script
    // We create a temporary .py file alongside the output or in /tmp
    std::string script_path = config.output_path + ".build_script.py";
    generate_python_script(config, script_path, platform);

    // 3. Execute Script
    std::string cmd = "python3 " + script_path;
    XINFER_LOG_INFO("Running RKNN Toolkit2 via Python...");

    int result = std::system(cmd.c_str());

    // 4. Cleanup
    fs::remove(script_path);

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("RKNN Model compiled successfully for " + platform);
        return true;
    } else {
        XINFER_LOG_ERROR("RKNN compilation failed. Return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler