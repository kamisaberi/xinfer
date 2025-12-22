#include <xinfer/compiler/drivers/vitis_driver.h>
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

bool VitisDriver::is_native_tool_available() {
    return (std::system("which vai_c_onnx > /dev/null 2>&1") == 0);
}

bool VitisDriver::validate_environment() {
    // 1. Check Native
    if (is_native_tool_available()) return true;

    // 2. Check Docker
    int res = std::system("docker --version > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_ERROR("Neither 'vai_c_onnx' nor 'docker' was found.");
        XINFER_LOG_WARN("Please install Docker or run this inside the Vitis AI container.");
        return false;
    }

    // Check if image exists
    res = std::system("docker image inspect xilinx/vitis-ai-cpu:latest > /dev/null 2>&1");
    if (res != 0) {
        XINFER_LOG_WARN("Docker image 'xilinx/vitis-ai-cpu:latest' not found. Pulling...");
        // Usually we let the user handle pulls, but we warn here.
    }

    return true;
}

bool VitisDriver::compile(const CompileConfig& config) {
    if (!fs::exists(config.input_path)) {
        XINFER_LOG_ERROR("Input file not found: " + config.input_path);
        return false;
    }

    // Vitis AI Compiler requires Quantized models (INT8)
    if (config.precision != Precision::INT8) {
        XINFER_LOG_WARN("Vitis AI DPU requires INT8 models. Ensure input ONNX is pre-quantized (QDQ).");
    }

    // --- Parse Architecture (Critical) ---
    // The compiler needs an arch.json file that matches the DPU on the FPGA.
    std::string arch_json;
    for (const auto& param : config.vendor_params) {
        if (param.find("DPU_ARCH=") == 0) {
            arch_json = param.substr(9);
        }
        else if (param.find("ARCH=") == 0) {
            arch_json = param.substr(5);
        }
    }

    if (arch_json.empty()) {
        XINFER_LOG_ERROR("Missing DPU Architecture file. Specify '--vendor-params DPU_ARCH=/path/to/arch.json'");
        return false;
    }

    // --- Prepare Paths ---
    // vai_c outputs to a directory. We use a temp dir then move the file.
    fs::path temp_out_dir = fs::temp_directory_path() / "xinfer_vitis_build";
    if (fs::exists(temp_out_dir)) fs::remove_all(temp_out_dir);
    fs::create_directories(temp_out_dir);

    std::string net_name = "xinfer_model";

    // --- Construct Command ---
    std::stringstream cmd;

    bool use_docker = !is_native_tool_available();

    if (use_docker) {
        // Map current directory to /workspace
        fs::path pwd = fs::current_path();
        cmd << "docker run --rm -v " << quote(pwd.string()) << ":/workspace ";
        cmd << "-w /workspace ";

        // If arch_json is absolute, we need to handle mapping, but assuming relative for CLI simplicity
        // or user handles the path mapping logic.
        // For robustness, we assume user runs CLI from project root.
        cmd << "xilinx/vitis-ai-cpu:latest ";
    }

    cmd << "vai_c_onnx";
    cmd << " --model " << quote(config.input_path);
    cmd << " --arch " << quote(arch_json);
    cmd << " --output_dir " << quote(temp_out_dir.string());
    cmd << " --net_name " << net_name;

    // --- Execute ---
    XINFER_LOG_INFO("Executing Vitis AI Compiler...");
    if (use_docker) XINFER_LOG_INFO("(Running inside Docker container)");
    XINFER_LOG_INFO("Cmd: " + cmd.str());

    int result = std::system(cmd.str().c_str());

    if (result != 0) {
        XINFER_LOG_ERROR("vai_c_onnx failed with return code: " + std::to_string(result));
        return false;
    }

    // --- Finalize Output ---
    // Vitis compiler creates: <output_dir>/<net_name>.xmodel
    fs::path generated_file = temp_out_dir / (net_name + ".xmodel");

    if (fs::exists(generated_file)) {
        try {
            // Move to user destination
            fs::path final_out(config.output_path);
            if (final_out.has_parent_path()) fs::create_directories(final_out.parent_path());

            fs::copy_file(generated_file, final_out, fs::copy_options::overwrite_existing);
            fs::remove_all(temp_out_dir);

            XINFER_LOG_SUCCESS("Vitis DPU .xmodel compiled successfully: " + config.output_path);
            return true;
        } catch (const std::exception& e) {
            XINFER_LOG_ERROR("Failed to move output file: " + std::string(e.what()));
            return false;
        }
    } else {
        XINFER_LOG_ERROR("Compiler finished, but output .xmodel not found.");
        return false;
    }
}

} // namespace xinfer::compiler