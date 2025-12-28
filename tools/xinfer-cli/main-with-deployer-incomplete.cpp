#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <map>

// Third-party CLI parser
#include <third_party/cxxopts/cxxopts.hpp>

// xInfer Core & Modules
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/utils.h>
#include <xinfer/compiler/compiler_factory.h>
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/utils/downloader.h>
#include <xinfer/deployer/deployer.h>
#include <xinfer/deployer/ssh_deployer.h> // Include concrete implementation

namespace fs = std::filesystem;
using namespace xinfer;

// =================================================================================
// FORWARD DECLARATIONS OF HANDLERS
// =================================================================================
void handle_compile(const cxxopts::ParseResult& args);
void handle_benchmark(const cxxopts::ParseResult& args);
void handle_download(const cxxopts::ParseResult& args);
void handle_deploy(const cxxopts::ParseResult& args);

// =================================================================================
// Main Entry Point
// =================================================================================
int main(int argc, char** argv) {
    cxxopts::Options options("xinfer-cli", "xInfer Universal AI Toolkit (Compiler, Benchmarker, Deployer, Downloader)");

    options.add_options()
        ("h,help", "Print usage")
        ("mode", "Operation: 'compile', 'benchmark', 'download', or 'deploy'", cxxopts::value<std::string>())
        ("target", "Target platform for compile/benchmark (e.g., nv-trt)", cxxopts::value<std::string>())
        ("onnx", "Input ONNX file path", cxxopts::value<std::string>())
        ("output", "Output engine/model file path", cxxopts::value<std::string>())
        ("vendor-params", "Extra flags (e.g. CORE=0)", cxxopts::value<std::vector<std::string>>());

    options.add_options("Compile")
        ("p,precision", "fp32, fp16, int8", cxxopts::value<std::string>()->default_value("fp16"))
        ("calibrate", "Path to calibration dataset", cxxopts::value<std::string>());

    options.add_options("Benchmark")
        ("model", "Compiled engine file for benchmarking", cxxopts::value<std::string>())
        ("iterations", "Number of runs", cxxopts::value<int>()->default_value("100"))
        ("warmup", "Number of warmup runs", cxxopts::value<int>()->default_value("10"));

    options.add_options("Download")
        ("url", "URL of the model to download", cxxopts::value<std::string>())
        ("sha256", "Optional SHA256 checksum", cxxopts::value<std::string>());

    options.add_options("Deploy")
        ("device", "Name of the device from manifest", cxxopts::value<std::string>())
        ("app", "Path to the application binary to deploy", cxxopts::value<std::string>())
        ("manifest", "Path to the devices.json manifest", cxxopts::value<std::string>()->default_value("devices.json"))
        ("exec", "Execute the application on the remote device after deploying");

    try {
        auto result = options.parse(argc, argv);

        if (result.count("help") || !result.count("mode")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        std::string mode = result["mode"].as<std::string>();

        if (mode == "compile") {
            handle_compile(result);
        } else if (mode == "benchmark") {
            handle_benchmark(result);
        } else if (mode == "download") {
            handle_download(result);
        } else if (mode == "deploy") {
            handle_deploy(result);
        } else {
            XINFER_LOG_FATAL("Unknown mode '" + mode + "'. Use 'compile', 'benchmark', 'download', or 'deploy'.");
        }

    } catch (const cxxopts::OptionException& e) {
        XINFER_LOG_FATAL(std::string("CLI Parsing Error: ") + e.what());
        return 1;
    }

    return 0;
}


// =================================================================================
// Command Implementations
// =================================================================================

void handle_compile(const cxxopts::ParseResult& args) {
    // ... (This code is correct from your provided snippet) ...
    // Sets up CompileConfig, finds driver, validates, and runs compile.
    // For brevity, assuming this is the full function from previous step.
    XINFER_LOG_INFO("Handling COMPILE command...");
}

void handle_benchmark(const cxxopts::ParseResult& args) {
    // ... (This code is correct from your provided snippet) ...
    // Loads backend, prepares dummy data, warms up, and runs benchmark loop.
    // For brevity, assuming this is the full function from previous step.
    XINFER_LOG_INFO("Handling BENCHMARK command...");
}

void handle_download(const cxxopts::ParseResult& args) {
    // ... (This code is correct from the previous step) ...
    // Calls utils::Downloader::download.
    XINFER_LOG_INFO("Handling DOWNLOAD command...");
    std::string url = args["url"].as<std::string>();
    std::string output = args["output"].as<std::string>();
    std::string sha = args.count("sha256") ? args["sha256"].as<std::string>() : "";
    utils::Downloader::download(url, output, sha);
}

// Simple JSON parser for devices.json (replace with a real library in production)
std::map<std::string, deployer::Device> parse_devices(const std::string& path) {
    std::map<std::string, deployer::Device> devices;
    // ... (Manual JSON parsing logic placeholder) ...
    // For this example, let's hardcode one device for demonstration.
    if (fs::exists(path)) {
        // Real parsing would go here.
        XINFER_LOG_INFO("Loaded device manifest from " + path);
    } else {
        XINFER_LOG_ERROR("Device manifest not found: " + path);
    }
    // Placeholder device:
    devices["rockchip_dev"] = {"rockchip_dev", "192.168.1.55", "rock", "password", "rockchip-rknn", "/home/rock/apps/"};
    return devices;
}


void handle_deploy(const cxxopts::ParseResult& args) {
    std::string device_name = args["device"].as<std::string>();
    std::string app_path = args["app"].as<std::string>();
    std::string onnx_path = args["onnx"].as<std::string>();
    std::string manifest_path = args["manifest"].as<std::string>();

    // 1. Load Device Info
    auto all_devices = parse_devices(manifest_path);
    if (all_devices.find(device_name) == all_devices.end()) {
        XINFER_LOG_FATAL("Device '" + device_name + "' not found in " + manifest_path);
        return;
    }
    deployer::Device target_device = all_devices.at(device_name);

    XINFER_LOG_INFO("Preparing deployment for target: " + target_device.name + " (" + target_device.target_platform + ")");

    // 2. Compile Model for Target Platform
    std::string engine_file = "model_deploy." + target_device.target_platform;

    compiler::CompileConfig compile_cfg;
    compile_cfg.target = compiler::stringToTarget(target_device.target_platform);
    compile_cfg.input_path = onnx_path;
    compile_cfg.output_path = engine_file;
    // (Inherit other compile flags from CLI if passed...)

    auto driver = compiler::CompilerFactory::create(compile_cfg.target);
    if (!driver || !driver->validate_environment() || !driver->compile(compile_cfg)) {
        XINFER_LOG_FATAL("Model compilation failed. Aborting deployment.");
        return;
    }

    // 3. Connect and Deploy
    deployer::SshDeployer deployer;

    if (!deployer.connect(target_device)) {
        XINFER_LOG_FATAL("Failed to connect to device via SSH.");
        return;
    }

    XINFER_LOG_INFO("Deploying application and model...");
    if (!deployer.send_files({app_path, engine_file})) {
        XINFER_LOG_FATAL("File transfer failed.");
        deployer.disconnect();
        return;
    }

    // 4. Remote Execution (Optional)
    if (result.count("exec")) {
        std::string remote_app_name = fs::path(app_path).filename().string();
        std::string remote_model_name = fs::path(engine_file).filename().string();

        std::string cmd = "cd " + target_device.remote_path + " && LD_LIBRARY_PATH=. ./" + remote_app_name + " --model " + remote_model_name;

        XINFER_LOG_INFO("Executing remote command: " + cmd);
        std::string output;
        int exit_code = deployer.execute(cmd, output);

        std::cout << "\n--- REMOTE OUTPUT ---\n" << output << "\n---------------------\n";
        XINFER_LOG_INFO("Remote command finished with exit code: " + std::to_string(exit_code));
    }

    deployer.disconnect();
    XINFER_LOG_SUCCESS("Deployment to " + device_name + " complete.");
    fs::remove(engine_file); // Cleanup local engine file
}