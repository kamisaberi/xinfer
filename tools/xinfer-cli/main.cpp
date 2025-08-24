// tools/xinfer-cli/main.cpp
#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <third_party/cxxopts/cxxopts.hpp>

#include <include/builders/engine_builder.h>
#include <include/core/engine.h>
#include <include/core/tensor.h>

bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

void handle_build(const cxxopts::ParseResult& args) {
    xinfer::builders::BuildFromUrlConfig config;
    bool is_url = args.count("from-url");

    if (is_url) {
        config.onnx_url = args["from-url"].as<std::string>();
    } else if (args.count("onnx")) {
        config.onnx_url = args["onnx"].as<std::string>();
        if (!file_exists(config.onnx_url)) {
            std::cerr << "Error: Local ONNX file not found at '" << config.onnx_url << "'" << std::endl;
            return;
        }
    } else {
        std::cerr << "Error: The 'build' command requires either --onnx (local file) or --from-url (web URL)." << std::endl;
        return;
    }

    config.output_engine_path = args["save_engine"].as<std::string>();
    if (args.count("fp16")) config.use_fp16 = true;
    if (args.count("int8")) config.use_int8 = true;
    if (args.count("batch")) config.max_batch_size = args["batch"].as<int>();

    std::cout << "Starting engine build process...\n";
    if (xinfer::builders::build_engine_from_url(config)) {
        std::cout << "\nSuccessfully built engine and saved to '" << config.output_engine_path << "'" << std::endl;
    } else {
        std::cerr << "\nEngine build process failed." << std::endl;
    }
}

void handle_benchmark(const cxxopts::ParseResult& args) {
    std::string engine_path = args["engine"].as<std::string>();
    int batch_size = args["batch"].as<int>();
    int iterations = args["iterations"].as<int>();

    if (!file_exists(engine_path)) {
        std::cerr << "Error: Engine file not found at '" << engine_path << "'" << std::endl;
        return;
    }

    std::cout << "Benchmarking engine '" << engine_path << "'...\n";
    std::cout << "  Batch Size: " << batch_size << "\n";
    std::cout << "  Iterations: " << iterations << "\n";

    try {
        xinfer::core::InferenceEngine engine(engine_path);
        auto input_shape = engine.get_input_shape(0);
        input_shape[0] = batch_size;
        xinfer::core::Tensor input_tensor(input_shape, xinfer::core::DataType::kFLOAT);

        std::cout << "  Warming up...\n";
        for (int i = 0; i < 20; ++i) { engine.infer({input_tensor}); }

        std::cout << "  Running benchmark...\n";
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) { engine.infer({input_tensor}); }
        auto end = std::chrono::high_resolution_clock::now();

        auto total_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
        double avg_time_ms = total_time_ms / iterations;

        std::cout << "\n--- Benchmark Results ---\n";
        printf("Total time for %d iterations: %.3f ms\n", iterations, total_time_ms);
        printf("Average inference latency: %.4f ms\n", avg_time_ms);
        printf("Throughput: %.2f inferences/sec\n", (iterations * batch_size) / (total_time_ms / 1000.0));

    } catch (const std::exception& e) {
        std::cerr << "\nError during benchmark: " << e.what() << std::endl;
    }
}

int main(int argc, char** argv) {
    cxxopts::Options options("xinfer-cli", "A command-line tool for the xInfer Performance Toolkit");

    options.add_options()
        ("h,help", "Print usage");

    options.add_options("Build")
        ("b,build", "Command to build an engine")
        ("onnx", "Path to the LOCAL input ONNX model file", cxxopts::value<std::string>())
        ("from-url", "URL to the remote ONNX model file", cxxopts::value<std::string>())
        ("save_engine", "Path to save the output TensorRT engine file", cxxopts::value<std::string>())
        ("fp16", "Enable FP16 precision mode")
        ("int8", "Enable INT8 precision mode (requires calibration)")
        ("batch", "Set the max batch size for the engine", cxxopts::value<int>()->default_value("1"))
        ("workspace", "Set the GPU workspace size in MB", cxxopts::value<size_t>());

    options.add_options("Benchmark")
        ("m,benchmark", "Command to benchmark an engine")
        ("engine", "Path to the TensorRT engine file to benchmark", cxxopts::value<std::string>())
        ("iterations", "Number of iterations to run", cxxopts::value<int>()->default_value("200"));

    if (argc == 1) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    auto result = options.parse(argc, argv);

    if (result.count("help")) {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result.count("build")) {
        if (!result.count("save_engine") || (!result.count("onnx") && !result.count("from-url"))) {
            std::cerr << "Error: The 'build' command requires --save_engine and either --onnx or --from-url." << std::endl;
            std::cout << options.help() << std::endl;
            return 1;
        }
        handle_build(result);
    } else if (result.count("benchmark")) {
        if (!result.count("engine")) {
            std::cerr << "Error: The 'benchmark' command requires the --engine argument." << std::endl;
            std::cout << options.help() << std::endl;
            return 1;
        }
        handle_benchmark(result);
    } else {
        std::cout << "No valid command specified.\n";
        std::cout << options.help() << std::endl;
    }

    return 0;
}