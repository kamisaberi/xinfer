#include <xinfer/compiler/drivers/intel_fpga_driver.h>
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

bool IntelFpgaDriver::validate_environment() {
    // 1. Check for OpenVINO Converter (ovc)
    int res = std::system("ovc --version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("OpenVINO 'ovc' tool not found.");
        return false;
    }

    // 2. Check for DLA Compiler
    // Usually located in the FPGA AI Suite installation
    const char* dla_root = std::getenv("INTEL_FPGA_AI_SUITE");
    if (dla_root) {
        fs::path compiler = fs::path(dla_root) / "bin" / "dla_compiler";
        if (fs::exists(compiler)) return true;
    }

    // Check system PATH as fallback
    res = std::system("dla_compiler --help > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("'dla_compiler' not found. Please install Intel FPGA AI Suite.");
        return false;
    }

    return true;
}

bool IntelFpgaDriver::convert_onnx_to_ir(const std::string& onnx_path,
                                         const std::string& out_dir,
                                         const std::string& model_name) {
    std::stringstream cmd;
    cmd << "ovc";
    cmd << " " << quote(onnx_path);
    cmd << " --output_model " << quote(out_dir + "/" + model_name + ".xml");
    // DLA usually prefers FP16
    cmd << " --compress_to_fp16";

    XINFER_LOG_INFO("Converting ONNX to OpenVINO IR...");
    return (std::system(cmd.str().c_str()) == 0);
}

bool IntelFpgaDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // Prepare temp directory for intermediate IR
    fs::path temp_dir = fs::temp_directory_path() / "xinfer_dla_build";
    if (fs::exists(temp_dir)) fs::remove_all(temp_dir);
    fs::create_directories(temp_dir);

    fs::path input_p(config.input_path);
    std::string stem = input_p.stem().string();

    // Step 1: Convert to OpenVINO IR if input is ONNX
    std::string ir_xml_path = config.input_path;

    if (input_p.extension() == ".onnx") {
        if (!convert_onnx_to_ir(config.input_path, temp_dir.string(), stem)) {
            XINFER_LOG_ERROR("Failed to convert ONNX to OpenVINO IR.");
            return false;
        }
        ir_xml_path = (temp_dir / (stem + ".xml")).string();
    }

    // Step 2: Run DLA Compiler
    // dla_compiler -network model.xml -arch arch.arch -o output.bin
    std::stringstream cmd;
    cmd << "dla_compiler";
    cmd << " -network " << quote(ir_xml_path);
    cmd << " -o " << quote(config.output_path);

    // --- Architecture File (Critical) ---
    // The compiler needs to know the FPGA resource config
    bool arch_found = false;
    for (const auto& param : config.vendor_params) {
        if (param.find("ARCH=") == 0) {
            cmd << " -arch " << quote(param.substr(5));
            arch_found = true;
        }
    }

    if (!arch_found) {
        // Fallback: Try to find a default .arch file in the SDK
        const char* dla_root = std::getenv("INTEL_FPGA_AI_SUITE");
        if (dla_root) {
            fs::path default_arch = fs::path(dla_root) / "arch" / "agilex_generic.arch";
            if (fs::exists(default_arch)) {
                XINFER_LOG_WARN("No ARCH specified. Using default: " + default_arch.string());
                cmd << " -arch " << quote(default_arch.string());
                arch_found = true;
            }
        }
    }

    if (!arch_found) {
        XINFER_LOG_ERROR("Missing Architecture file. Please specify '--vendor-params ARCH=/path/to/dla.arch'");
        return false;
    }

    XINFER_LOG_INFO("Running Intel DLA Compiler...");
    int res = std::system(cmd.str().c_str());

    // Cleanup
    fs::remove_all(temp_dir);

    if (res == 0 && fs::exists(config.output_path)) {
        XINFER_LOG_SUCCESS("Intel FPGA DLA Binary compiled successfully.");
        return true;
    } else {
        XINFER_LOG_ERROR("dla_compiler failed.");
        return false;
    }
}

} // namespace xinfer::compiler