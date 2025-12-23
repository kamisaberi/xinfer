#pragma once
#include <xinfer/core/tensor.h>
#include "types.h"

namespace xinfer::preproc {

    struct ImageFrame {
        void* data;         // Raw pointer (Host or Device)
        int width;
        int height;
        ImageFormat format; // RGB, BGR, NV12
        bool is_device_ptr; // Is this already on GPU/NPU memory?
    };

    class IImagePreprocessor {
    public:
        virtual ~IImagePreprocessor() = default;

        // The main function: Raw Frame -> Network Tensor
        virtual void process(const ImageFrame& src, core::Tensor& dst) = 0;
    };

}