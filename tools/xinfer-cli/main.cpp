#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <memory>

// Third-party
#include <third_party/cxxopts/cxxopts.hpp>

// xInfer Modern Core & Compiler Extensions
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>
#include <xinfer/core/device_manager.h>
#include <xinfer/compiler/compiler_factory.h>
#include <xinfer/backends/backend_factory.h>

namespace fs = std::filesystem;

/**
 * @brief Handles the 'compile' command (replaces old build)
 * This handles the ONNX -> [Target Engine] conversion for all 15 platforms.
 */
void handle_compile(const cxxopts::ParseResult& args) {
    using namespace xinfer::compiler;

    CompileConfig config;
    
    // 1. Resolve Target Platform
    std::string target_str = args["target"].as<std::string>();
    config.target = stringToTarget(target_str);

    // 2. Set Model Sources (Local or URL)
    if (args.count("onnx")) {
        config.input_path = args["onnx"].as<std::string>();
    } else if (args.count("from-url")) {
        config.onnx_url = args["from-url"].as<std::string>();
    } else {
        XINFER_LOG_ERROR("Compilation requires --onnx or --from-url");
        return;
    }

    // 3. Configure Precision & Optimization
    config.output_path = args["output"].as<std::string>();
    config.precision = stringToPrecision(args["precision"].as<std::string>());
    
    if (args.count("calibrate")) {
        config.calibration_data_path = args["calibrate"].as<std::string>();
    }

    // 4. Capture Extra Vendor Params (e.g., DPU Arch for FPGA, HTP for Qualcomm)
    if (args.count("vendor-params")) {
        config.vendor_params = args["vendor-params"].as<std::vector<std::string>>();
    }

    // 5. Dispatch to Factory
    XINFER_LOG_INFO("Initializing compiler driver for: " + target_str);
    auto compiler = CompilerFactory::create(config.target);

    if (!compiler->validate_environment()) {
        XINFER_LOG_FATAL("Environment check failed for " + target_str + ". Run 'xinfer-doctor' to fix.");
        return;
    }

    XINFER_LOG_INFO("Starting compilation process...");
    auto start = std::chrono::high_resolution_clock::now();
    
    if (compiler->compile(config)) {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        XINFER_LOG_SUCCESS("Engine built successfully in " + std::to_string(duration) + "s -> " + config.output_path);
    } else {
        XINFER_LOG_ERROR("Compilation failed for target: " + target_str);
    }
}

/**
 * @brief Handles the 'benchmark' command
 * Generalizes benchmarking to work across any backend runtime.
 */
void handle_benchmark(const cxxopts::ParseResult& args) {
    std::string engine_path = args["engine"].as<std::string>();
    int batch_size = args["batch"].as<int>();
    int iterations = args["iterations"].as<int>();
    std::string target_str = args["target"].as<std::string>();

    XINFER_LOG_INFO("Benchmarking [" + target_str + "] engine: " + engine_path);

    try {
        // Create the runtime backend based on the target
        auto backend = xinfer::backends::BackendFactory::create(stringToTarget(target_str));
        backend->load_model(engine_path);

        // Prepare dummy input (Generalizing based on backend metadata)
        auto input_meta = backend->get_input_metadata(0);
        xinfer::core::Tensor input_tensor(input_meta.shape, input_meta.dtype);

        XINFER_LOG_INFO("Warming up...");
        for (int i = 0; i < 10; ++i) { backend->predict({input_tensor}); }

        XINFER_LOG_INFO("Running " + std::to_string(iterations) + " iterations...");
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            backend->predict({input_tensor});
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto total_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        
        std::cout << "\n--- Performance Report (" << target_str << ") ---\n";
        printf("Latency (Avg): %.4f ms\n", total_ms / iterations);
        printf("Throughput:    %.2f FPS\n", (iterations * batch_size) / (total_ms / 1000.0));

    } catch (const std::exception& e) {
        XINFER_LOG_ERROR("Benchmark failed: " + std::string(e.what()));
    }
}

int main(int argc, char** argv) {
    cxxopts::Options options("xinfer-cli", "xInfer Universal Performance Toolkit CLI");

    options.add_options()
        ("h,help", "Print usage")
        ("v,version", "Print version info");

    // Unified Compile Options
    options.add_options("Compile")
        ("c,compile", "Trigger compilation mode")
        ("target", "Target platform (nv-trt, amd-vitis, rk-npu, hailo, qcom-qnn, etc.)", cxxopts::value<std::string>())
        ("onnx", "Local path to ONNX model", cxxopts::value<std::string>())
        ("from-url", "Remote URL for ONNX model", cxxopts::value<std::string>())
        ("output", "Output engine file path", cxxopts::value<std::string>())
        ("precision", "Bit-precision (fp32, fp16, int8, int4)", cxxopts::value<std::string>()->default_value("fp16"))
        ("calibrate", "Path to calibration data for INT8", cxxopts::value<std::string>())
        ("vendor-params", "Extra vendor flags (key=value)", cxxopts::value<std::vector<std::string>>());

    // Unified Benchmark Options
    options.add_options("Benchmark")
        ("b,benchmark", "Trigger benchmark mode")
        ("engine", "Path to the compiled engine file", cxxopts::value<std::string>())
        ("batch", "Inference batch size", cxxopts::value<int>()->default_value("1"))
        ("iterations", "Number of iterations", cxxopts::value<int>()->default_value("100"));

    if (argc == 1) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    // Route to Command Handlers
    if (result.count("compile")) {
        handle_compile(result);
    } else if (result.count("benchmark")) {
        handle_benchmark(result);
    } else {
        XINFER_LOG_WARN("No command specified. Use --compile or --benchmark.");
        std::cout << options.help() << std::endl;
    }

    return 0;
}