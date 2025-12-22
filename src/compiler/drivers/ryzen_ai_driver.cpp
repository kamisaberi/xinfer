#include <xinfer/compiler/drivers/ryzen_ai_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <filesystem>

namespace fs = std::filesystem;

namespace xinfer::compiler {

bool RyzenAIDriver::validate_environment() {
    // Check if vai_q_onnx is installed in the current Python env
    // This is usually part of the Ryzen AI Software conda environment.
    int res = std::system("python -c \"import vai_q_onnx\" > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Python module 'vai_q_onnx' not found.");
        XINFER_LOG_WARN("Please install Ryzen AI Software and activate the Conda environment.");
        XINFER_LOG_WARN("Command: pip install vai_q_onnx");
        return false;
    }
    return true;
}

std::string RyzenAIDriver::generate_quantize_script(const CompileConfig& config,
                                                    const std::string& script_path) {
    std::ofstream py(script_path);

    py << "import sys\n";
    py << "import vai_q_onnx\n";
    py << "import numpy as np\n";
    py << "from onnxruntime.quantization import CalibrationDataReader\n\n";

    py << "input_model_path = '" << config.input_path << "'\n";
    py << "output_model_path = '" << config.output_path << "'\n";

    // --- Calibration Data Reader Class ---
    py << "class XInferDataReader(CalibrationDataReader):\n";
    py << "    def __init__(self, npy_path):\n";
    py << "        self.data = np.load(npy_path)\n";
    py << "        # If data is (N, C, H, W), we iterate one by one\n";
    py << "        self.enum_data = iter([{ 'input': x[np.newaxis, :] } for x in self.data])\n\n"; // Assumes model input name is 'input', genericizer needed for prod
    py << "    def get_next(self):\n";
    py << "        return next(self.enum_data, None)\n\n";

    // --- Main Logic ---
    py << "try:\n";
    py << "    print('[RyzenAI] Starting Quantization...')\n";

    // 1. Setup Data Reader
    if (!config.calibration_data_path.empty()) {
        py << "    # Using provided .npy calibration data\n";
        py << "    dr = XInferDataReader('" << config.calibration_data_path << "')\n";
    } else {
        // Fallback: Random Data (Not recommended for accuracy, but allows pipeline test)
        py << "    print('WARNING: No calibration data provided. Using random noise (Accuracy will be poor).')\n";
        py << "    # Create dummy data: 10 frames of 1x3x640x640 (generic guess)\n";
        py << "    dummy = np.random.rand(10, 3, 640, 640).astype(np.float32)\n";
        py << "    np.save('temp_dummy_calib.npy', dummy)\n";
        py << "    dr = XInferDataReader('temp_dummy_calib.npy')\n";
    }

    // 2. Run Quantize Static
    // Ryzen AI NPU requires "QDQ" format (Quantize-Dequantize nodes)
    py << "    vai_q_onnx.quantize_static(\n";
    py << "        input_model_path,\n";
    py << "        output_model_path,\n";
    py << "        calibration_data_reader=dr,\n";
    py << "        quant_format=vai_q_onnx.QuantFormat.QDQ,\n"; // Crucial for XDNA
    py << "        calibrate_method=vai_q_onnx.PowerOfTwoMethod.MinMSE,\n"; // PowerOfTwo preferred for FPGA/NPU
    py << "        activation_type=vai_q_onnx.QuantType.QUInt8,\n";
    py << "        weight_type=vai_q_onnx.QuantType.QInt8,\n";
    py << "        enable_dpu=True,\n";  // Optimize for DPU/NPU
    py << "        extra_options={'ActivationSymmetric': True}\n"; // XDNA requirement
    py << "    )\n";

    py << "    print(f'[RyzenAI] Quantized model saved to {output_model_path}')\n";

    py << "except Exception as e:\n";
    py << "    print(f'Error during quantization: {e}')\n";
    py << "    sys.exit(1)\n";

    py.close();
    return script_path;
}

bool RyzenAIDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // Ryzen AI strictly prefers INT8
    if (config.precision != Precision::INT8) {
        XINFER_LOG_WARN("Ryzen AI NPU requires INT8 quantization for acceleration.");
        XINFER_LOG_WARN("You requested " + std::to_string((int)config.precision) + ". Proceeding, but performace may vary.");
    }

    // 1. Generate Python Script
    std::string script_path = config.output_path + ".quantize.py";
    generate_quantize_script(config, script_path);

    // 2. Execute
    std::string cmd = "python " + script_path;
    XINFER_LOG_INFO("Executing Vitis AI Quantizer...");

    int result = std::system(cmd.c_str());

    // 3. Cleanup
    fs::remove(script_path);
    if (fs::exists("temp_dummy_calib.npy")) fs::remove("temp_dummy_calib.npy");

    if (result == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("Ryzen AI Model prepared successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("vai_q_onnx failed.");
        return false;
    }
}

} // namespace xinfer::compiler