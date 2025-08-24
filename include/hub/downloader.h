#pragma once

#include <string>
#include <vector>
#include "model_info.h"

namespace xinfer::hub {

    /**
     * @brief Lists all models available from the hub.
     * @param hub_url The base URL of your model hub.
     * @return A vector of ModelInfo structs describing the available models.
     */
    std::vector<ModelInfo> list_models(const std::string& hub_url = "https://api.your-ignition-hub.com");

    /**
     * @brief Downloads a pre-built TensorRT engine file from the hub.
     *
     * This function handles downloading, caching, and verifying the engine file.
     *
     * @param model_id The unique identifier of the model (e.g., "yolov8n-coco").
     * @param target The desired hardware and precision target.
     * @param cache_dir The local directory to store downloaded engine files.
     * @param hub_url The base URL of your model hub.
     * @return The local file path to the downloaded .engine file.
     */
    std::string download_engine(const std::string& model_id,
                              const HardwareTarget& target,
                              const std::string& cache_dir = "./xinfer_cache",
                              const std::string& hub_url = "https://api.your-ignition-hub.com");

} // namespace xinfer::hub
