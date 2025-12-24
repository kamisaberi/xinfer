#pragma once

#include <xinfer/core/tensor.h>
#include <vector>
#include <array>

namespace xinfer::postproc {

/**
 * @brief 3D Bounding Box
 * Represents an oriented box in 3D space (LiDAR coordinate system).
 * Center (x,y,z), Dimensions (w,l,h), Rotation (yaw).
 */
struct BoundingBox3D {
    float x, y, z;       // Center Position (meters)
    float w, l, h;       // Dimensions (width, length, height)
    float yaw;           // Rotation around Z-axis (radians)
    float velocity_x;    // Optional: for tracking (CenterPoint)
    float velocity_y;    
    float score;
    int class_id;
};

/**
 * @brief Configuration for 3D Detection
 */
struct Detection3DConfig {
    float score_threshold = 0.3f;
    float nms_threshold = 0.1f; // 3D IoU threshold is usually lower than 2D
    int max_detections = 500;

    // Grid Configuration (Crucial for decoding)
    // How big is one pixel in the output feature map in real meters?
    float voxel_size_x = 0.1f; 
    float voxel_size_y = 0.1f;

    // Point Cloud Range [min_x, min_y, min_z, max_x, max_y, max_z]
    // Used to calculate the origin offset.
    std::array<float, 6> pc_range = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};

    // Output stride (Feature map downscaling factor)
    int downsample_ratio = 1; 
};

class IDetection3DPostprocessor {
public:
    virtual ~IDetection3DPostprocessor() = default;

    virtual void init(const Detection3DConfig& config) = 0;

    /**
     * @brief Process raw 3D detection heads.
     * 
     * Expects a specific layout, usually:
     * 1. Heatmap (Scores) [1, NumClasses, Y, X]
     * 2. Regression Map (Pos/Dim/Rot) [1, RegChannels, Y, X]
     * 
     * @param tensors Vector of output tensors from the model.
     * @return List of 3D Boxes.
     */
    virtual std::vector<BoundingBox3D> process(const std::vector<core::Tensor>& tensors) = 0;
};

} // namespace xinfer::postproc