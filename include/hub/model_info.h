#pragma once

#include <string>
#include <vector>

namespace xinfer::hub {

    struct HardwareTarget {
        std::string gpu_architecture; // e.g., "RTX_4090", "Jetson_Orin_Nano"
        std::string tensorrt_version; // e.g., "10.1.0"
        std::string precision;        // "FP32", "FP16", "INT8"
    };

    struct ModelInfo {
        std::string model_id;         // e.g., "yolov8n-coco"
        std::string task;             // e.g., "object-detection"
        std::string framework;        // "xInfer-TensorRT"
        std::vector<HardwareTarget> available_targets;
    };

} // namespace xinfer::hub
