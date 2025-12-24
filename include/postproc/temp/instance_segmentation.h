
#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <include/core/tensor.h>

namespace xinfer::postproc::instance_segmentation {

    // The final, processed result for a single instance
    struct InstanceSegmentationResult {
        int class_id;
        float confidence;
        cv::Rect bounding_box;
        cv::Mat mask; // Binary mask, same size as original image
    };

    /**
     * @brief Processes the raw outputs of an instance segmentation model.
     *
     * This is a high-level function that orchestrates a complex, multi-stage
     * post-processing pipeline entirely on the GPU for maximum performance. It handles
     * detection decoding, NMS, mask coefficient multiplication, and mask resizing.
     *
     * @param raw_detections GPU tensor containing [boxes, mask_coeffs, scores, classes].
     * @param mask_prototypes GPU tensor containing the mask prototypes.
     * @param conf_thresh Confidence score threshold for detections.
     * @param nms_thresh IoU threshold for Non-Maximum Suppression.
     * @param mask_thresh Binary threshold for the final instance masks.
     * @param model_input_width The width the model was run at.
     * @param model_input_height The height the model was run at.
     * @param original_image_width The width of the original user image.
     * @param original_image_height The height of the original user image.
     * @return A vector of processed InstanceSegmentationResult structs on the CPU.
     */
    std::vector<InstanceSegmentationResult> process(
        const core::Tensor& raw_detections,
        const core::Tensor& mask_prototypes,
        float conf_thresh,
        float nms_thresh,
        float mask_thresh,
        int model_input_width,
        int model_input_height,
        int original_image_width,
        int original_image_height
    );

} // namespace xinfer::postproc::instance_segmentation

