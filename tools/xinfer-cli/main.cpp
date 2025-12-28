#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <filesystem>

// Third-party CLI parser
#include <third_party/cxxopts/cxxopts.hpp>

// xInfer Core & Compiler
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/utils.h>
#include <xinfer/compiler/compiler_factory.h>
#include <xinfer/compiler/base_compiler.h> // For stringToTarget
#include <xinfer/backends/backend_factory.h>


#include <xinfer/utils/downloader.h> // <-- Add this include



namespace fs = std::filesystem;
using namespace xinfer;

// =================================================================================
// Helper: Parse "Key=Value" strings from CLI
// =================================================================================
std::vector<std::string> parse_vector_args(const std::vector<std::string>& raw_args) {
    return raw_args; // The Compiler Driver handles parsing specific "KEY=VAL" logic
}

// =================================================================================
// Command: COMPILE
// =================================================================================
void handle_compile(const cxxopts::ParseResult& args) {
    std::string target_str = args["target"].as<std::string>();
    std::string input_path;

    // Validate Input
    if (args.count("onnx")) input_path = args["onnx"].as<std::string>();
    else if (args.count("tflite")) input_path = args["tflite"].as<std::string>();
    else {
        XINFER_LOG_FATAL("Compile requires input model (--onnx or --tflite)");
        return;
    }

    if (!fs::exists(input_path)) {
        XINFER_LOG_FATAL("Input file does not exist: " + input_path);
        return;
    }

    // 1. Configure
    compiler::CompileConfig config;
    try {
        config.target = compiler::stringToTarget(target_str);
    } catch (const std::exception& e) {
        XINFER_LOG_FATAL(e.what());
        return;
    }

    config.input_path = input_path;
    config.output_path = args["output"].as<std::string>();

    // Precision parsing
    std::string prec_str = args["precision"].as<std::string>();
    config.precision = compiler::stringToPrecision(prec_str);

    // Calibration data
    if (args.count("calibrate")) {
        config.calibration_data_path = args["calibrate"].as<std::string>();
    }

    // Vendor flags (e.g. "CORE=0", "ARCH=hailo8")
    if (args.count("vendor-params")) {
        config.vendor_params = args["vendor-params"].as<std::vector<std::string>>();
    }

    // 2. Create Driver
    auto driver = compiler::CompilerFactory::create(config.target);
    if (!driver) {
        XINFER_LOG_FATAL("Failed to create compiler driver for target: " + target_str);
        return;
    }

    XINFER_LOG_INFO("Selected Driver: " + driver->get_name());

    // 3. Validate Environment
    if (!driver->validate_environment()) {
        XINFER_LOG_FATAL("Environment validation failed. Run 'xinfer-doctor' for details.");
        return;
    }

    // 4. Run Compilation
    auto start = std::chrono::high_resolution_clock::now();
    XINFER_LOG_INFO("Starting compilation...");

    bool success = driver->compile(config);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0;

    if (success) {
        XINFER_LOG_SUCCESS("Build Complete in " + std::to_string(duration) + "s");
        XINFER_LOG_INFO("Output saved to: " + config.output_path);
    } else {
        XINFER_LOG_FATAL("Compilation Failed.");
    }
}

// =================================================================================
// Command: BENCHMARK
// =================================================================================
void handle_benchmark(const cxxopts::ParseResult& args) {
    std::string model_path = args["model"].as<std::string>();
    std::string target_str = args["target"].as<std::string>();
    int iterations = args["iterations"].as<int>();
    int warmup = args["warmup"].as<int>();

    if (!fs::exists(model_path)) {
        XINFER_LOG_FATAL("Model file not found: " + model_path);
        return;
    }

    // 1. Load Backend
    Target target;
    try {
        target = compiler::stringToTarget(target_str);
    } catch (...) {
        XINFER_LOG_FATAL("Unknown target: " + target_str);
        return;
    }

    XINFER_LOG_INFO("Loading backend for " + target_str + "...");

    // Config for runtime
    Config runtime_config;
    runtime_config.model_path = model_path;
    // Pass vendor params to runtime if needed (e.g. "DEVICE_ID=0")
    if (args.count("vendor-params")) {
        runtime_config.vendor_params = args["vendor-params"].as<std::vector<std::string>>();
    }

    auto engine = backends::BackendFactory::create(target); // Note: Factory usually needs config or uses load_model later
    // *Assuming BackendFactory returns a unique_ptr and we call load separately or factory takes config*
    // Based on previous designs, we register lambda that takes config.
    // Let's assume standard factory usage:

    if (!engine || !engine->load_model(model_path)) {
        XINFER_LOG_FATAL("Failed to load model on backend.");
        return;
    }

    XINFER_LOG_INFO("Model loaded on: " + engine->device_name());

    // 2. Prepare Dummy Data
    // We need to inspect model input shapes.
    // *Assuming IBackend has a method like get_input_info() or we use fixed size for test*
    // For this CLI tool, we create a generic tensor. Real impl would query backend.
    std::vector<core::Tensor> inputs;
    std::vector<core::Tensor> outputs;

    // Create a dummy 640x640x3 tensor (Batch 1) as placeholder
    core::Tensor dummy_input({1, 3, 640, 640}, core::DataType::kFLOAT);
    float* ptr = static_cast<float*>(dummy_input.data());
    std::fill(ptr, ptr + dummy_input.size(), 0.5f); // Fill gray
    inputs.push_back(dummy_input);

    // 3. Warmup
    XINFER_LOG_INFO("Warming up (" + std::to_string(warmup) + " iters)...");
    for (int i = 0; i < warmup; ++i) {
        engine->predict(inputs, outputs);
    }

    // 4. Benchmark Loop
    XINFER_LOG_INFO("Benchmarking (" + std::to_string(iterations) + " iters)...");
    std::vector<double> times_ms;
    times_ms.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        engine->predict(inputs, outputs);

        // Ensure sync for timing (some backends are async)
        // If backend exposes synchronize(), call it.
        // Or assume predict blocks or add a sync mechanism.
        // For CLI accuracy, explicit sync is best if available.

        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        times_ms.push_back(ms);
    }

    // 5. Stats
    double sum = std::accumulate(times_ms.begin(), times_ms.end(), 0.0);
    double mean = sum / times_ms.size();

    std::sort(times_ms.begin(), times_ms.end());
    double min = times_ms.front();
    double max = times_ms.back();
    double p99 = times_ms[(int)(times_ms.size() * 0.99)];

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << " BENCHMARK RESULTS: " << fs::path(model_path).filename().string() << std::endl;
    std::cout << " Target: " << engine->device_name() << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;
    printf(" Average Latency:  %.3f ms\n", mean);
    printf(" Min Latency:      %.3f ms\n", min);
    printf(" Max Latency:      %.3f ms\n", max);
    printf(" P99 Latency:      %.3f ms\n", p99);
    printf(" Throughput (B=1): %.2f FPS\n", 1000.0 / mean);
    std::cout << "--------------------------------------------------" << std::endl;
}


// =================================================================================
// Command: DOWNLOAD
// =================================================================================
void handle_download(const cxxopts::ParseResult& args) {
    std::string url = args["url"].as<std::string>();
    std::string output = args["output"].as<std::string>();
    std::string sha = args.count("sha256") ? args["sha256"].as<std::string>() : "";

    utils::Downloader::download(url, output, sha);
}


// =================================================================================
// Main Entry
// =================================================================================
int main(int argc, char** argv) {
    cxxopts::Options options("xinfer-cli", "Universal AI Compiler & Benchmark Tool");

    options.add_options()
        ("h,help", "Print usage")
        ("mode", "Operation mode: 'compile' or 'benchmark'", cxxopts::value<std::string>())
        ("target", "Target Platform (nv-trt, rockchip-rknn, amd-vitis, etc.)", cxxopts::value<std::string>())

        // Compile Options
        ("onnx", "Input ONNX file path", cxxopts::value<std::string>())
        ("tflite", "Input TFLite file path", cxxopts::value<std::string>())
        ("o,output", "Output engine file path", cxxopts::value<std::string>())
        ("p,precision", "fp32, fp16, int8", cxxopts::value<std::string>()->default_value("fp16"))
        ("calibrate", "Path to calibration dataset (txt/npy)", cxxopts::value<std::string>())

        // Benchmark Options
        ("model", "Compiled engine file for benchmarking", cxxopts::value<std::string>())
        ("iterations", "Number of runs", cxxopts::value<int>()->default_value("100"))
        ("warmup", "Number of warmup runs", cxxopts::value<int>()->default_value("10"))
        
        // Common
        ("vendor-params", "Extra flags (e.g. CORE=0, DPU_ARCH=x.json)", cxxopts::value<std::vector<std::string>>());


    options.add_options("Download")
        ("download", "Trigger download mode")
        ("url", "URL of the model to download", cxxopts::value<std::string>())
        // Re-use 'output' from compile options
        ("sha256", "Optional SHA256 checksum for verification", cxxopts::value<std::string>());

        
    try {
        auto result = options.parse(argc, argv);

        if (result.count("help") || !result.count("mode")) {
            std::cout << options.help() << std::endl;
            return 0;
        }

        // --- Update your main logic ---
        if (result.count("download")) {
            handle_download(result);
        } else if (mode == "compile") {
            // ...
        } // ...

        std::string mode = result["mode"].as<std::string>();

        if (mode == "compile") {
            handle_compile(result);
        } else if (mode == "benchmark") {
            handle_benchmark(result);
        } else {
            XINFER_LOG_FATAL("Unknown mode. Use 'compile' or 'benchmark'.");
        }

    } catch (const cxxopts::OptionException& e) {
        XINFER_LOG_FATAL(std::string("CLI Parsing Error: ") + e.what());
        return 1;
    }

    return 0;
}