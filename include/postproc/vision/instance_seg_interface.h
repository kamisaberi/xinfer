#pragma once

#include <xinfer/postproc/vision/instance_seg_interface.h>
#include <xinfer/postproc/vision/types.h>
#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CPU Implementation of Instance Segmentation (YOLO-Seg).
 * 
 * Target: Rockchip NPU, Intel CPU, Mobile.
 * 
 * Workflow:
 * 1. Decode Boxes & Mask Coefficients from Output 0.
 * 2. Perform NMS to filter down to ~10-20 detections.
 * 3. Matrix Multiply: (Surviving Coefficients) x (Prototype Masks).
 * 4. Sigmoid Activation -> Crop to Box -> Resize to Original Image.
 */
class CpuInstanceSegPostproc : public IInstanceSegmentationPostprocessor {
public:
    CpuInstanceSegPostproc();
    ~CpuInstanceSegPostproc() override;

    void init(const InstanceSegConfig& config) override;

    /**
     * @brief Process raw outputs.
     * 
     * Expects:
     * - Tensor[0]: Detection Head [Batch, 4 + Cls + 32, Anchors]
     * - Tensor[1]: Proto Head [Batch, 32, 160, 160] (Mask Prototypes)
     */
    std::vector<InstanceResult> process(const std::vector<core::Tensor>& tensors) override;

private:
    InstanceSegConfig m_config;

    struct RawDetection {
        float x1, y1, x2, y2;
        float score;
        int class_id;
        std::vector<float> mask_coeffs; // Usually 32 floats
    };

    /**
     * @brief Decodes the YOLOv8-Seg format.
     */
    void decode_yolov8_seg(const float* det_data, 
                           const std::vector<int64_t>& det_shape,
                           std::vector<RawDetection>& proposals);

    /**
     * @brief Generates final masks for valid detections.
     * Uses optimized matrix multiplication.
     */
    void resolve_masks(const std::vector<RawDetection>& detections,
                       const float* proto_data,
                       const std::vector<int64_t>& proto_shape,
                       std::vector<InstanceResult>& results);
};

} // namespace xinfer::postproc