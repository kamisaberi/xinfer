#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <cxxopts.hpp>

// Include the high-level builder and core engine from your xinfer library
#include <xinfer/builders/engine_builder.h>
#include <xinfer/core/engine.h>
#include <xinfer/core/tensor.h>

// Helper to check for file existence
bool file_exists(const std::string& path) {
    std::ifstream f(path.c_str());
    return f.good();
}

/**
 * @brief Handles the "build" command.
 * Converts an ONNX file to an optimized TensorRT engine.
 */
void handle_build(const cxxopts::ParseResult& args) {
    std::string onnx_path = args["onnx"].as<std::string>();
    std::string engine_path = args["save_engine"].as<std::string>();

    if (!file_exists(onnx_path)) {
        std::cerr << "Error: ONNX file not found at '" << onnx_path << "'" << std::endl;
        return;
    }

    std::cout << "Building TensorRT engine for '" << onnx_path << "'..." << std::endl;
    std::cout << "This may take several minutes." << std::endl;

    try {
        xinfer::builders::EngineBuilder builder;
        builder.from_onnx(onnx_path);

        if (args.count("fp16")) {
            std::cout << "  -> Enabling FP16 precision." << std::endl;
            builder.with_fp16();
        }
        if (args.count("int8")) {
            std::cout << "  -> Enabling INT8 precision." << std::endl;
            // Note: A real implementation needs a Calibrator object here.
            // For this CLI, we assume a simple case or a pre-calibrated model.
            // builder.with_int8(...);
            std::cout << "  (Warning: INT8 calibration not fully implemented in this example CLI)\n";
        }
        if (args.count("batch")) {
            int batch = args["batch"].as<int>();
            std::cout << "  -> Setting max batch size to " << batch << "." << std::endl;
            builder.with_max_batch_size(batch);
        }
        if (args.count("workspace")) {
            size_t workspace = args["workspace"].as<size_t>();
            std::cout << "  -> Setting workspace size to " << workspace << " MB." << std::endl;
            // builder.with_workspace_size(workspace); // Assuming this method exists
        }

        builder.build_and_save(engine_path);

        std::cout << "\nSuccessfully built engine and saved to '" << engine_path << "'" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "\nError during engine build: " << e.what() << std::endl;
    }
}

/**
 * @brief Handles the "benchmark" command.
 * Runs performance tests on a pre-built engine.
 */
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

        // Create dummy input data
        auto input_shape = engine.get_input_shape(0);
        input_shape[0] = batch_size; // Adjust batch size
        xinfer::core::Tensor input_tensor(input_shape, xinfer::core::DataType::kFLOAT);
        // In a real benchmark, you might fill this with random data.

        // Warm-up
        std::cout << "  Warming up...\n";
        for (int i = 0; i < 20; ++i) {
            engine.infer({input_tensor});
        }

        std::cout << "  Running benchmark...\n";
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            engine.infer({input_tensor});
        }
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
        ("onnx", "Path to the input ONNX model file", cxxopts::value<std::string>())
        ("save_engine", "Path to save the output TensorRT engine file", cxxopts::value<std::string>())
        ("fp16", "Enable FP16 precision mode")
        ("int8", "Enable INT8 precision mode (requires calibration)")
        ("batch", "Set the max batch size for the engine", cxxopts::value<int>()->default_value("1"))
        ("workspace", "Set the GPU workspace size in MB", cxxopts::value<size_t>());

    options.add_options("Benchmark")
        ("m,benchmark", "Command to benchmark an engine")
        ("engine", "Path to the TensorRT engine file to benchmark", cxxopts::value<std::string>())
        ("iterations", "Number of iterations to run", cxxopts::value<int>()->default_value("200"));

    // Check if a command was provided
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
        if (!result.count("onnx") || !result.count("save_engine")) {
            std::cerr << "Error: The 'build' command requires both --onnx and --save_engine arguments." << std::endl;
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