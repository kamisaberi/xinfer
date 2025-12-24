#pragma once

#include <memory>
#include <xinfer/compiler/base_compiler.h> // For 'Target' enum

// --- Vision Interfaces ---
#include <xinfer/postproc/vision/detection_interface.h>
#include <xinfer/postproc/vision/segmentation_interface.h>
#include <xinfer/postproc/vision/instance_seg_interface.h>
#include <xinfer/postproc/vision/anomaly_interface.h>
#include <xinfer/postproc/vision/detection3d_interface.h>

// --- Text Interfaces ---
#include <xinfer/postproc/text/ocr_interface.h>

// --- Generative Interfaces ---
#include <xinfer/postproc/generative/sampler_interface.h>

namespace xinfer::postproc {

/**
 * @brief Factory for Post-Processing Modules.
 *
 * Automatically selects the most efficient implementation based on the
 * Target hardware.
 *
 * Logic:
 * - If Target == NVIDIA_TRT -> Returns CUDA implementations (keeps data on GPU).
 * - If Target == APPLE_COREML -> Returns Metal implementations (if available).
 * - Else -> Returns optimized CPU implementations (OpenCV/NEON/AVX).
 */

// --- Vision ---

/**
 * @brief Creates a Detection Post-processor (YOLO, SSD).
 * Performs decoding and NMS.
 */
std::unique_ptr<IDetectionPostprocessor> create_detection(xinfer::Target target);

/**
 * @brief Creates a Semantic Segmentation Post-processor.
 * Performs ArgMax and resizing.
 */
std::unique_ptr<ISegmentationPostprocessor> create_segmentation(xinfer::Target target);

/**
 * @brief Creates an Instance Segmentation Post-processor.
 * Decodes masks and bounding boxes (Mask R-CNN / YOLO-Seg).
 */
std::unique_ptr<IInstanceSegmentationPostprocessor> create_instance_segmentation(xinfer::Target target);

/**
 * @brief Creates an Anomaly Detection Post-processor.
 * Calculates heatmaps/MSE between input and reconstruction.
 */
std::unique_ptr<IAnomalyPostprocessor> create_anomaly(xinfer::Target target);

/**
 * @brief Creates a 3D Detection Post-processor (LiDAR/PointPillars).
 */
std::unique_ptr<IDetection3DPostprocessor> create_detection3d(xinfer::Target target);


// --- Text ---

/**
 * @brief Creates an OCR/CTC Decoder.
 * Decodes probabilities into strings (Greedy/Beam Search).
 */
std::unique_ptr<IOcrPostprocessor> create_ocr(xinfer::Target target);


// --- Generative ---

/**
 * @brief Creates a Diffusion Sampler.
 * Handles noise scheduling and sampling steps.
 */
std::unique_ptr<ISamplerPostprocessor> create_sampler(xinfer::Target target);

} // namespace xinfer::postproc