#pragma once

#include <xinfer/postproc/vision/detection3d_interface.h>

namespace xinfer::postproc {

/**
 * @brief CPU Implementation of 3D Detection Post-processing.
 * 
 * Target: CenterPoint / PointPillars architectures.
 * 
 * Optimization:
 * - Uses vectorized pointers to scan the Heatmap efficiently.
 * - Implements a custom BEV (Bird's Eye View) NMS for rotated boxes.
 */
class CpuDetection3DPostproc : public IDetection3DPostprocessor {
public:
    CpuDetection3DPostproc();
    ~CpuDetection3DPostproc() override;

    void init(const Detection3DConfig& config) override;

    std::vector<BoundingBox3D> process(const std::vector<core::Tensor>& tensors) override;

private:
    Detection3DConfig m_config;

    // Derived constants
    float m_min_x, m_min_y;
    float m_out_size_factor_x;
    float m_out_size_factor_y;

    /**
     * @brief Custom NMS for Rotated Boxes (BEV).
     * Standard OpenCV NMS is axis-aligned only. This handles rotation approximation.
     */
    void apply_nms_bev(std::vector<BoundingBox3D>& boxes, float thresh);
};

} // namespace xinfer::postproc