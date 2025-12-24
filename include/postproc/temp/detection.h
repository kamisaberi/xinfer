#pragma once

#include <vector>
#include "../core/tensor.h"

namespace xinfer::postproc {

    struct BoundingBox { /* x1, y1, x2, y2, class_id, confidence, ... */ };

    /**
     * @brief Performs Non-Maximum Suppression (NMS) on raw detection outputs on the GPU.
     * @param raw_boxes A GPU tensor of shape [N, 4] with box coordinates.
     * @param raw_scores A GPU tensor of shape [N, num_classes] with scores.
     * @param score_threshold The confidence score threshold.
     * @param iou_threshold The IoU threshold for suppression.
     * @return A vector of BoundingBox structs, copied back to the CPU.
     */
    std::vector<BoundingBox> nms(const core::Tensor& raw_boxes,
                                 const core::Tensor& raw_scores,
                                 float score_threshold,
                                 float iou_threshold);

} // namespace xinfer::postproc
