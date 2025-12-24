#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"
#include <vector>

namespace xinfer::postproc {

    class IDetectionPostprocessor {
    public:
        virtual ~IDetectionPostprocessor() = default;

        // Configure thresholds
        virtual void init(const DetectionConfig& config) = 0;

        /**
         * @brief Decodes model output into bounding boxes.
         * @param tensors The raw output from the inference engine.
         *                (YOLO often produces 1 or 3 output tensors).
         */
        virtual std::vector<BoundingBox> process(const std::vector<core::Tensor>& tensors) = 0;
    };

}