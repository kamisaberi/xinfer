#include <iostream>
#include <filesystem>
#include <csignal>

// xInfer Flow Header
#include <xinfer/flow/pipeline.h>
#include <xinfer/core/logging.h>

using namespace xinfer;

// Global flag for clean exit on Ctrl+C
static bool keep_running = true;

void signal_handler(int) {
    keep_running = false;
}

int main(int argc, char** argv) {
    // 1. Setup
    if (argc < 2) {
        std::cerr << "Usage: ./run_pipeline <path_to_pipeline.json>" << std::endl;
        return -1;
    }

    std::string config_path = argv[1];
    if (!std::filesystem::exists(config_path)) {
        XINFER_LOG_FATAL("Config file not found: " + config_path);
        return -1;
    }

    // Handle Ctrl+C
    std::signal(SIGINT, signal_handler);

    // 2. Initialize Pipeline
    std::cout << "Loading Pipeline from: " << config_path << "..." << std::endl;

    flow::Pipeline pipe;
    if (!pipe.load(config_path)) {
        XINFER_LOG_FATAL("Failed to load pipeline.");
        return -1;
    }

    // 3. Run
    // pipe.run() blocks until EOS or error.
    // Since we want to handle Ctrl+C to stop gracefully,
    // we can run it in a thread or check if the API supports non-blocking.
    // For this example, we assume run() handles its own loop checking.

    std::cout << "Pipeline running. Press Ctrl+C to stop." << std::endl;

    // In a real app, you might run this in a std::thread
    pipe.run();

    // 4. Cleanup
    // (Destructor handles it)
    std::cout << "Pipeline stopped." << std::endl;
    return 0;
}