#pragma once

#include <xinfer/postproc/vision/types.h> // BoundingBox struct
#include <vector>
#include <memory>

namespace xinfer::postproc {

/**
 * @brief Metal-Accelerated NMS (Non-Maximum Suppression)
 * 
 * Optimized for Apple Silicon (M1/M2/M3) and iOS.
 * 
 * Strategy:
 * - Uses Unified Memory (Zero-Copy) to share box data.
 * - Computes the N*N IoU matrix using a massive parallel Metal Kernel.
 * - Suitable when number of boxes > 2000, otherwise CPU NMS is usually faster.
 */
class MetalNMS {
public:
    MetalNMS();
    ~MetalNMS();

    /**
     * @brief Run NMS on the provided boxes.
     * 
     * @param boxes Input list of boxes (unsorted).
     * @param iou_threshold Intersection over Union threshold (e.g., 0.45).
     * @param max_output_boxes Maximum number of boxes to return.
     * @return Indices of the boxes to keep.
     */
    std::vector<int> process(const std::vector<BoundingBox>& boxes, 
                             float iou_threshold, 
                             int max_output_boxes);

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} // namespace xinfer::postproc