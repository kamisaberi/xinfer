#pragma once

#include <xinfer/postproc/vision/instance_seg_interface.h>
#include <vector>

namespace xinfer::postproc {

/**
 * @brief CPU Implementation of Instance Segmentation.
 * 
 * Optimized for Rockchip, Intel, and Mobile CPUs.
 * Uses OpenCV for SIMD-accelerated Matrix Multiplication (GEMM) 
 * to combine Mask Coefficients with Prototypes efficiently.
 */
class CpuInstanceSegPostproc : public IInstanceSegmentationPostprocessor {
public:
    CpuInstanceSegPostproc();
    ~CpuInstanceSegPostproc() override;

    void init(const InstanceSegConfig& config) override;

    std::vector<InstanceResult> process(const std::vector<core::Tensor>& tensors) override;

private:
    InstanceSegConfig m_config;

    // Internal structure to hold data before NMS
    struct RawDetection {
        float x1, y1, x2, y2;
        float score;
        int class_id;
        std::vector<float> mask_coeffs; // The weights for the linear combination
    };

    /**
     * @brief Decodes the YOLOv8-Seg output format.
     * Output shape is typically [Batch, Channels, Anchors].
     */
    void decode_yolo_seg(const float* det_data, 
                         const std::vector<int64_t>& det_shape,
                         std::vector<RawDetection>& proposals);

    /**
     * @brief Generates masks for the surviving detections.
     * 
     * Performs: (Coeffs x Proto) -> Sigmoid -> Resize -> Crop
     */
    void resolve_masks(const std::vector<RawDetection>& detections,
                       const float* proto_data,
                       const std::vector<int64_t>& proto_shape,
                       std::vector<InstanceResult>& results);
};

} // namespace xinfer::postproc