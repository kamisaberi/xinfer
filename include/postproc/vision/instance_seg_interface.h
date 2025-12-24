#pragma once

#include <xinfer/core/tensor.h>
#include <xinfer/postproc/vision/types.h> // Contains BoundingBox struct
#include <vector>

namespace xinfer::postproc {

/**
 * @brief Result for Instance Segmentation
 * Contains the Bounding Box and the Pixel Mask.
 */
struct InstanceResult {
    BoundingBox box;
    
    // Binary Mask (uint8). 
    // Shape: [1, ImageHeight, ImageWidth]
    // 0 = Background, 255 = Object
    core::Tensor mask; 
};

/**
 * @brief Configuration for Instance Segmentation
 */
struct InstanceSegConfig {
    float conf_threshold = 0.25f;
    float nms_threshold = 0.45f;
    int max_detections = 100;
    
    // YOLO Specifics
    int num_classes = 80;
    int num_mask_protos = 32; // Standard for YOLOv5/v8-seg
    
    // Target Image Dimensions (Required to resize masks correctly)
    int target_width = 640;
    int target_height = 640;
};

/**
 * @brief Interface for Instance Segmentation Post-processors.
 */
class IInstanceSegmentationPostprocessor {
public:
    virtual ~IInstanceSegmentationPostprocessor() = default;

    virtual void init(const InstanceSegConfig& config) = 0;

    /**
     * @brief Process raw model outputs.
     * 
     * Expects 2 Input Tensors:
     * 1. Detection Head: [Batch, 4 + NumClasses + NumProtos, NumAnchors]
     * 2. Proto Head:     [Batch, NumProtos, MaskH, MaskW]
     */
    virtual std::vector<InstanceResult> process(const std::vector<core::Tensor>& tensors) = 0;
};

} // namespace xinfer::postproc