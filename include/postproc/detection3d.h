#pragma once


#include <include/core/tensor.h>
#include <vector>

namespace xinfer::postproc::detection3d {

    /**
     * @brief Performs 3D Non-Maximum Suppression on raw detection outputs on the GPU.
     *
     * This function takes GPU tensors of decoded 3D bounding boxes and scores,
     * and efficiently filters them using a 3D IoU (Intersection over Union)
     * calculation. The entire process runs on the GPU for maximum performance.
     *
     * @param decoded_boxes A GPU tensor of shape [NumBoxes, 7] where each row is
     *                      [center_x, center_y, center_z, length, width, height, yaw].
     * @param decoded_scores A GPU tensor of shape [NumBoxes] with the confidence score for each box.
     * @param nms_iou_threshold The 3D IoU threshold for suppression. Boxes with an IoU
     *                          greater than this value will be suppressed.
     * @return A std::vector<int> containing the indices of the boxes that survived suppression.
     *         The indices correspond to the rows in the input tensors.
     */
    std::vector<int> nms(
        const core::Tensor& decoded_boxes,
        const core::Tensor& decoded_scores,
        float nms_iou_threshold
    );

} // namespace xinfer::postproc::detection3d

