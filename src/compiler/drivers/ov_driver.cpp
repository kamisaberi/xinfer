#include <xinfer/compiler/drivers/ov_driver.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/utils.h>

#include <iostream>
#include <sstream>
#include <filesystem>
#include <cstdlib>

namespace fs = std::filesystem;

namespace xinfer::compiler {

static std::string quote(const std::string& path) {
    return "\"" + path + "\"";
}

bool OpenVINODriver::validate_environment() {
    // Check if 'ovc' is in the system PATH
    // (Part of the openvino-dev Python package)
    int res = std::system("ovc --version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Tool 'ovc' not found.");
        XINFER_LOG_WARN("Please install OpenVINO Dev Tools: pip install openvino-dev");
        return false;
    }
    return true;
}

bool OpenVINODriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    std::stringstream cmd;
    cmd << "ovc";

    // --- Input / Output ---
    cmd << " " << quote(config.input_path);

    // OpenVINO output requires the .xml extension
    fs::path out_p(config.output_path);
    if (out_p.extension() != ".xml") {
        XINFER_LOG_WARN("OpenVINO output should end in .xml. Appending extension.");
        out_p.replace_extension(".xml");
    }
    cmd << " --output_model " << quote(out_p.string());

    // --- Precision ---
    // OpenVINO 2024+ defaults to FP16 for IR generation if compressed
    if (config.precision == Precision::FP16) {
        cmd << " --compress_to_fp16";
    }
    else if (config.precision == Precision::INT8) {
        // NOTE: Standard 'ovc' does not perform Quantization (PTQ).
        // That requires NNCF (Neural Network Compression Framework).
        // If the user provided a pre-quantized ONNX (QAT), ovc will preserve it.
        XINFER_LOG_INFO("INT8 requested. Assuming input ONNX is already quantized (FakeQuant).");
        // If not pre-quantized, we warn.
        XINFER_LOG_WARN("To perform Post-Training Quantization, use NNCF tools externally.");
    }

    // --- Vendor Specific Params ---
    // Handle things like input shapes or layouts
    for (const auto& param : config.vendor_params) {
        // Example: "SHAPE=[1,3,640,640]"
        if (param.find("SHAPE=") == 0) {
            cmd << " --input_shape " << param.substr(6);
        }
        // Example: "LAYOUT=nhwc"
        else if (param.find("LAYOUT=") == 0) {
             cmd << " --layout " << param.substr(7);
        }
        // Raw flags
        else {
             cmd << " " << param;
        }
    }

    // --- Execution ---
    XINFER_LOG_INFO("Executing OpenVINO Converter: " + cmd.str());

    int result = std::system(cmd.str().c_str());

    // Check for both .xml and .bin
    fs::path bin_p = out_p;
    bin_p.replace_extension(".bin");

    if (result == 0 && fs::exists(out_p) && fs::exists(bin_p)) {
        XINFER_LOG_SUCCESS("OpenVINO IR generated successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("ovc failed with return code: " + std::to_string(result));
        return false;
    }
}

} // namespace xinfer::compiler