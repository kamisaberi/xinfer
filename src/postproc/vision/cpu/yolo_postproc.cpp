#include "yolo_postproc.h"
#include <xinfer/core/logging.h>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp> // Optimized NMSBoxes
#include <algorithm>
#include <cmath>

namespace xinfer::postproc {

// =================================================================================
// Constructor / Init
// =================================================================================

CpuYoloPostproc::CpuYoloPostproc() {}
CpuYoloPostproc::~CpuYoloPostproc() {}

void CpuYoloPostproc::init(const DetectionConfig& config) {
    m_config = config;
}

// =================================================================================
// Main Process Dispatcher
// =================================================================================

std::vector<BoundingBox> CpuYoloPostproc::process(const std::vector<core::Tensor>& tensors) {
    std::vector<BoundingBox> proposals;
    if (tensors.empty()) return proposals;

    // 1. Determine Model Architecture based on Output Shape
    // YOLOv8/v9/v10 typically output [Batch, 4 + NumClasses, NumAnchors] (e.g., [1, 84, 8400])
    // YOLOv5/v7 typically output [Batch, NumAnchors, 5 + NumClasses] (e.g., [1, 25200, 85])
    
    // We assume Batch Size = 1 for this implementation. 
    // If Batch > 1, we process the first image or loop (omitted for brevity).
    const auto& output = tensors[0];
    auto shape = output.shape(); // e.g., {1, 84, 8400}

    // Heuristic: Check the dimensions
    // If dim[1] is small (~84) and dim[2] is large (~8400), it's YOLOv8 Transposed
    // If dim[1] is large (~25200) and dim[2] is small (~85), it's YOLOv5 Standard
    
    bool is_yolov8_layout = (shape[1] < shape[2]); 

    if (is_yolov8_layout) {
        // Output: [1, Channels, Anchors] -> Needs strided access
        decode_yolov8(static_cast<const float*>(output.data()), shape, proposals);
    } else {
        // Output: [1, Anchors, Channels] -> Linear access
        decode_yolov5({output}, proposals);
    }

    // 2. Perform Non-Maximum Suppression (NMS)
    // We use OpenCV's NMSBoxes which uses AVX/NEON optimization internally.
    std::vector<cv::Rect> boxes_cv;
    std::vector<float> confidences_cv;
    
    boxes_cv.reserve(proposals.size());
    confidences_cv.reserve(proposals.size());

    for (const auto& p : proposals) {
        boxes_cv.emplace_back((int)p.x1, (int)p.y1, (int)(p.x2 - p.x1), (int)(p.y2 - p.y1));
        confidences_cv.push_back(p.confidence);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes_cv, confidences_cv, m_config.conf_threshold, m_config.nms_threshold, indices);

    // 3. Limit detections
    if (indices.size() > m_config.max_detections) {
        indices.resize(m_config.max_detections);
    }

    // 4. Pack Final Results
    std::vector<BoundingBox> final_results;
    final_results.reserve(indices.size());

    for (int idx : indices) {
        final_results.push_back(proposals[idx]);
    }

    return final_results;
}

// =================================================================================
// Logic: YOLOv8 / Anchor-Free Decoding
// =================================================================================

void CpuYoloPostproc::decode_yolov8(const float* data, 
                                    const std::vector<int64_t>& shape, 
                                    std::vector<BoundingBox>& proposals) {
    // Shape is [1, NumClasses + 4, NumAnchors]
    // e.g. [1, 84, 8400] for COCO
    int num_classes = (int)shape[1] - 4;
    int num_anchors = (int)shape[2];
    
    // Pointers to the start of rows
    // Row 0: cx, Row 1: cy, Row 2: w, Row 3: h
    const float* p_cx = data;
    const float* p_cy = data + num_anchors;
    const float* p_w  = data + num_anchors * 2;
    const float* p_h  = data + num_anchors * 3;
    
    // Class scores start at Row 4
    const float* p_scores = data + num_anchors * 4;

    // Loop over all anchors
    for (int i = 0; i < num_anchors; ++i) {
        
        // Find max class score
        float max_score = 0.0f;
        int class_id = -1;

        // Iterate classes for this anchor (Strided by num_anchors)
        for (int c = 0; c < num_classes; ++c) {
            float score = p_scores[c * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                class_id = c;
            }
        }

        if (max_score >= m_config.conf_threshold) {
            // Extract Box
            float cx = p_cx[i];
            float cy = p_cy[i];
            float w  = p_w[i];
            float h  = p_h[i];

            // Convert Center-WH to TopLeft-BottomRight
            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            proposals.push_back({x1, y1, x2, y2, max_score, class_id});
        }
    }
}

// =================================================================================
// Logic: YOLOv5 / v7 / Anchor-Based Decoding
// =================================================================================

void CpuYoloPostproc::decode_yolov5(const std::vector<core::Tensor>& tensors, 
                                    std::vector<BoundingBox>& proposals) {
    // Usually flattened output: [1, NumAnchors, 5 + NumClasses]
    // Or 3 output layers. Here we assume the flattened single tensor format 
    // (standard export from Ultralytics).
    
    const auto& tensor = tensors[0];
    const float* data = static_cast<const float*>(tensor.data());
    
    int num_anchors = tensor.shape()[1];
    int stride = tensor.shape()[2]; // 5 + NumClasses
    int num_classes = stride - 5;

    for (int i = 0; i < num_anchors; ++i) {
        const float* row = data + (i * stride);
        
        float obj_conf = row[4];
        if (obj_conf < m_config.conf_threshold) continue;

        // Find max class probability
        float max_class_score = 0.0f;
        int class_id = -1;

        for (int c = 0; c < num_classes; ++c) {
            float class_score = row[5 + c];
            if (class_score > max_class_score) {
                max_class_score = class_score;
                class_id = c;
            }
        }

        // Final confidence = Objectness * ClassProb
        float final_conf = obj_conf * max_class_score;

        if (final_conf >= m_config.conf_threshold) {
            float cx = row[0];
            float cy = row[1];
            float w  = row[2];
            float h  = row[3];

            float x1 = cx - w * 0.5f;
            float y1 = cy - h * 0.5f;
            float x2 = cx + w * 0.5f;
            float y2 = cy + h * 0.5f;

            proposals.push_back({x1, y1, x2, y2, final_conf, class_id});
        }
    }
}

} // namespace xinfer::postproc