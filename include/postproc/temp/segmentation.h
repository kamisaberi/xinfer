#pragma once

#include <vector>
#include <opencv2/opencv.hpp> // For returning cv::Mat, a common format for masks
#include "../core/tensor.h"

namespace xinfer::postproc {

/**
 * @brief Performs an ArgMax operation across the channel dimension of a segmentation output tensor.
 *
 * This is the fundamental post-processing step for semantic segmentation. It converts the
 * raw logits tensor of shape [Batch, NumClasses, Height, Width] into a 2D integer
 * tensor of shape [Batch, Height, Width], where each pixel's value is the class index
 * with the highest score.
 *
 * This entire operation is performed in a single, efficient CUDA kernel.
 *
 * @param logits A GPU tensor (typically FP32 or FP16) containing the raw output from a segmentation model.
 * @param output_mask A pre-allocated GPU tensor (typically INT32) to store the resulting class masks.
 */
void argmax(const core::Tensor& logits, core::Tensor& output_mask);


/**
 * @brief A convenience function that performs argmax and downloads the result to a cv::Mat.
 *
 * This function is designed for ease of use. It wraps the `argmax` function and handles
 * the GPU-to-CPU data transfer, returning a standard OpenCV matrix that can be
 * immediately used for visualization or further processing.
 *
 * @param logits A GPU tensor containing the raw segmentation output.
 * @return A cv::Mat of type CV_32S (32-bit integer), where each pixel value is the predicted class ID.
 *         Note: Assumes a batch size of 1.
 */
cv::Mat argmax_to_mat(const core::Tensor& logits);


/**
 * @brief Applies a color map to a semantic segmentation mask on the GPU.
 *
 * This function takes a single-channel integer mask (the output of `argmax`) and
 * converts it into a 3-channel BGR color image for visualization. This is much faster
 * than downloading the mask to the CPU and using OpenCV to apply a color map.
 *
 * @param mask A single-channel GPU tensor of type INT32 containing class IDs.
 * @param color_map A GPU tensor of shape [NumClasses, 3] containing the BGR color values.
 * @param output_image A pre-allocated 3-channel GPU tensor (e.g., UC8) to store the colored mask.
 */
void apply_color_map(const core::Tensor& mask,
                     const core::Tensor& color_map,
                     core::Tensor& output_image);

// --------------------------------------------------------------------------
//                  Functions for Instance Segmentation
// --------------------------------------------------------------------------

// A struct to hold the results for a single detected instance
struct InstanceMask {
    int class_id;
    float confidence;
    // Bounding box of the instance
    int x1, y1, x2, y2;
    // A binary (0 or 1) mask for this specific instance, same size as the original image
    cv::Mat mask;
};

/**
 * @brief Processes the raw outputs of an instance segmentation model like Mask R-CNN.
 *
 * This is a complex, high-level function that would internally use multiple custom
 * CUDA kernels to perform the full post-processing pipeline on the GPU:
 * 1. Applies score thresholds to raw detections.
 * 2. Performs Non-Maximum Suppression (NMS) on the bounding boxes.
 * 3. For the surviving detections, it crops and thresholds the raw mask logits.
 * 4. Resizes the final masks to the original image dimensions.
 *
 * @param raw_boxes A GPU tensor of shape [NumDetections, 4] with box coordinates.
 * @param raw_scores A GPU tensor of shape [NumDetections, NumClasses] with scores.
 * @param raw_masks A GPU tensor of shape [NumDetections, MaskHeight, MaskWidth] with mask logits.
 * @param score_threshold The confidence score threshold.
 * @param iou_threshold The IoU threshold for NMS.
 * @param original_image_width The width of the original input image.
 * @param original_image_height The height of the original input image.
 * @return A vector of InstanceMask structs, with final data copied to the CPU.
 */
std::vector<InstanceMask> process_instance_masks(const core::Tensor& raw_boxes,
                                                 const core::Tensor& raw_scores,
                                                 const core::Tensor& raw_masks,
                                                 float score_threshold,
                                                 float iou_threshold,
                                                 int original_image_width,
                                                 int original_image_height);

} // namespace xinfer::postproc
