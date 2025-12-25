#pragma once

#include <memory>
#include <xinfer/compiler/base_compiler.h> // For 'Target' enum

// --- Vision Interfaces ---
#include <xinfer/postproc/vision/detection_interface.h>
#include <xinfer/postproc/vision/segmentation_interface.h>
#include <xinfer/postproc/vision/instance_seg_interface.h>
#include <xinfer/postproc/vision/anomaly_interface.h>
#include <xinfer/postproc/vision/detection3d_interface.h>
#include <xinfer/postproc/vision/classification_interface.h>
#include <xinfer/postproc/vision/tracker_interface.h>

// --- Text Interfaces ---
#include <xinfer/postproc/text/ocr_interface.h>
#include <xinfer/postproc/text/llm_sampler_interface.h>

// --- Generative Interfaces ---
#include <xinfer/postproc/generative/sampler_interface.h>

namespace xinfer::postproc {

/**
 * @brief Post-Processing Factory
 *
 * Central switchboard for creating hardware-accelerated post-processors.
 *
 * Logic:
 * - Checks the requested Target (NVIDIA, Rockchip, CPU, etc.).
 * - Returns the specialized implementation if available (e.g., CUDA).
 * - Falls back to optimized CPU implementation (OpenCV/AVX/NEON) otherwise.
 */

// ================= Vision =================

// YOLO / SSD Decoding & NMS
std::unique_ptr<IDetectionPostprocessor> create_detection(xinfer::Target target);

// Semantic Segmentation (ArgMax + Resize)
std::unique_ptr<ISegmentationPostprocessor> create_segmentation(xinfer::Target target);

// Instance Segmentation (Mask Assembly)
std::unique_ptr<IInstanceSegmentationPostprocessor> create_instance_segmentation(xinfer::Target target);

// Anomaly Detection (Reconstruction Error)
std::unique_ptr<IAnomalyPostprocessor> create_anomaly(xinfer::Target target);

// 3D Object Detection (Lidar/PointPillars)
std::unique_ptr<IDetection3DPostprocessor> create_detection3d(xinfer::Target target);

// Classification (Top-K)
std::unique_ptr<IClassificationPostprocessor> create_classification(xinfer::Target target);

// Object Tracking (Kalman/SORT)
std::unique_ptr<ITracker> create_tracker(xinfer::Target target);


// ================= Text =================

// OCR / Speech Decoding (CTC)
std::unique_ptr<IOcrPostprocessor> create_ocr(xinfer::Target target);

// LLM Token Sampling (Top-P/Top-K)
std::unique_ptr<ILlmSampler> create_llm_sampler(xinfer::Target target);


// ================= Generative =================

// Diffusion Model Sampling (DDIM)
std::unique_ptr<ISamplerPostprocessor> create_sampler(xinfer::Target target);

} // namespace xinfer::postproc