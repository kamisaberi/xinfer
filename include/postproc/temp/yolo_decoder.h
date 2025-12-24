#pragma once

#include <include/core/tensor.h>

namespace xinfer::postproc::yolo {

    /**
     * @brief Decodes the raw output of a YOLO-style model on the GPU.
     *
     * This function takes the single, large output tensor from a model like YOLOv8
     * and efficiently converts it into three separate tensors: one for bounding box
     * coordinates, one for confidence scores, and one for class IDs.
     *
     * @param raw_output The raw GPU tensor from the model.
     * @param confidence_threshold A threshold to filter out low-confidence boxes on the GPU.
     * @param out_boxes A pre-allocated GPU tensor to store the [x1, y1, x2, y2] coordinates.
     * @param out_scores A pre-allocated GPU tensor to store the confidence scores.
     * @param out_classes A pre-allocated GPU tensor to store the class IDs.
     */
    void decode(const core::Tensor& raw_output,
                float confidence_threshold,
                core::Tensor& out_boxes,
                core::Tensor& out_scores,
                core::Tensor& out_classes);

} // namespace xinfer::postproc::yolo

