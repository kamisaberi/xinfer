#pragma once

#include <xinfer/postproc/vision/detection_interface.h>
#include <xinfer/postproc/vision/types.h>
#include <xinfer/core/tensor.h>
#include <vector>

namespace xinfer::postproc {

    /**
     * @brief CPU Implementation of YOLO Post-processing.
     *
     * Optimized for ARM (Rockchip/Mobile) and x86 (Intel/AMD) CPUs.
     * It handles:
     * 1. Output decoding (Sigmoid/Softmax on raw pointers).
     * 2. Coordinate transformation (Grid -> Absolute).
     * 3. Non-Maximum Suppression (NMS) using optimized algorithms.
     */
    class CpuYoloPostproc : public IDetectionPostprocessor {
    public:
        CpuYoloPostproc();
        ~CpuYoloPostproc() override;

        /**
         * @brief Initialize thresholds and anchor configurations.
         */
        void init(const DetectionConfig& config) override;

        /**
         * @brief Decodes raw YOLO output tensors into bounding boxes.
         *
         * Handles inputs from NPUs (which map tensor memory to CPU address space).
         *
         * @param tensors The raw output tensors from the backend.
         *                (YOLOv8 usually has 1 tensor, YOLOv5 has 3).
         * @return List of valid bounding boxes.
         */
        std::vector<BoundingBox> process(const std::vector<core::Tensor>& tensors) override;

    private:
        DetectionConfig m_config;

        /**
         * @brief Helper to decode YOLOv8 (Anchor-free) format.
         * Output shape: [Batch, 4 + NumClasses, NumAnchors] (often transposed)
         */
        void decode_yolov8(const float* data,
                           const std::vector<int64_t>& shape,
                           std::vector<BoundingBox>& proposals);

        /**
         * @brief Helper to decode YOLOv5/v7 (Anchor-based) format.
         * Output shape: [Batch, NumAnchors, GridY, GridX, 5 + NumClasses]
         */
        void decode_yolov5(const std::vector<core::Tensor>& tensors,
                           std::vector<BoundingBox>& proposals);
    };

} // namespace xinfer::postproc