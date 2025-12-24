#pragma once
#include <vector>

namespace xinfer::postproc {

    struct BoundingBox {
        float x1, y1, x2, y2;
        float confidence;
        int class_id;
        // Optional: for 3D
        float z = 0.0f, w = 0.0f, h = 0.0f, l = 0.0f, rot_y = 0.0f;
    };

    struct DetectionConfig {
        float conf_threshold = 0.25f;
        float nms_threshold = 0.45f;
        int max_detections = 1000;
        std::vector<float> anchors; // For YOLO
        int num_classes = 80;
    };

}