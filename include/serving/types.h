#pragma once

#include <string>
#include <vector>
#include <map>
#include <cstdint>

namespace xinfer::serving {

    /**
     * @brief Server Configuration
     */
    struct ServerConfig {
        std::string host = "0.0.0.0";
        int port = 8080;

        // Path to the directory containing model files (.engine, .rknn, .onnx)
        std::string model_repository_path;

        // Thread pool size for handling concurrent requests
        int num_threads = 4;

        // Enable detailed request logging
        bool enable_logging = true;
    };

    /**
     * @brief Standard Inference Request format (JSON-compatible)
     */
    struct InferenceRequest {
        std::string model_name;
        std::vector<int64_t> input_shape;
        std::vector<float> input_data;

        // Optional: Request specific version or device
        std::string version = "latest";
        std::string device_id = "";
    };

    /**
     * @brief Standard Inference Response format
     */
    struct InferenceResponse {
        std::string model_name;
        std::vector<int64_t> output_shape;
        std::vector<float> output_data;
        double inference_time_ms;

        bool success;
        std::string error_message;
    };

} // namespace xinfer::serving