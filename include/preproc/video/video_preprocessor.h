#pragma once
#include <xinfer/core/tensor.h>
#include <xinfer/preproc/image/image_preprocessor.h> // Reuses ImageFrame

namespace xinfer::preproc {

    struct VideoConfig {
        int time_steps = 16;   // Number of frames to stack (T)
        int height = 224;
        int width = 224;
        int channels = 3;
        bool layout_ntchw = true; // [Batch, Time, Channels, Height, Width]
    };

    class IVideoPreprocessor {
    public:
        virtual ~IVideoPreprocessor() = default;
        virtual void init(const VideoConfig& config) = 0;

        /**
         * @brief Pushes a new frame into the buffer and retrieves the full stack.
         *
         * @param new_frame The single current image from the camera.
         * @param dst_stack Output 5D Tensor (Batch=1, T, C, H, W).
         */
        virtual void push_and_get(const ImageFrame& new_frame, core::Tensor& dst_stack) = 0;

        /**
         * @brief Clears history (e.g., when switching tracking targets).
         */
        virtual void reset() = 0;
    };

}