#include <xinfer/serving/server.h>
#include <xinfer/core/logging.h>
#include <iostream>
#include <filesystem>

// Simple signal handler for Ctrl+C
#include <csignal>
#include <atomic>

std::atomic<bool> stop_requested(false);

void signal_handler(int) {
    stop_requested = true;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./xinfer-server <model_repository_path> [port]" << std::endl;
        return 1;
    }

    std::string repo_path = argv[1];
    int port = (argc > 2) ? std::stoi(argv[2]) : 8080;

    if (!std::filesystem::exists(repo_path)) {
        std::cerr << "Error: Directory does not exist: " << repo_path << std::endl;
        return 1;
    }

    // Configure Server
    xinfer::serving::ServerConfig config;
    config.port = port;
    config.model_repo_path = repo_path;
    config.num_threads = 8; // Adjust based on CPU cores

    xinfer::serving::ModelServer server(config);

    std::cout << "----------------------------------------" << std::endl;
    std::cout << " xInfer Model Server Running" << std::endl;
    std::cout << " Port: " << port << std::endl;
    std::cout << " Repo: " << repo_path << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Run in a separate thread so we can handle exit signals,
    // or just let it block main thread.
    std::signal(SIGINT, signal_handler);

    // Note: server.start() is blocking.
    // In a real app, you might run it in a std::thread and join it.
    server.start();

    return 0;
}