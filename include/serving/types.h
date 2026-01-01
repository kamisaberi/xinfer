#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace xinfer::serving {

    struct ServerConfig {
        std::string host = "0.0.0.0";
        int port = 8080;
        int num_threads = 4;
        std::string model_repo_path; // Directory containing .engine, .rknn, .xml files
    };

    // Request body: { "input": [...], "shape": [1, 3, 224, 224] }
    struct InferenceRequest {
        std::vector<float> input_data;
        std::vector<int64_t> shape;
    };

    // Response body: { "output": [...], "time_ms": 12.5 }
    struct InferenceResponse {
        std::vector<float> output_data;
        std::vector<int64_t> output_shape;
        double inference_time_ms;
    };

}