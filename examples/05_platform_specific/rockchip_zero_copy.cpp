#include <iostream>
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/core/tensor.h>

// Mocking a DRM/DMA Buffer allocator (platform specific logic)
int alloc_dma_buffer(size_t size) { return 42; /* fd */ }
void* map_dma_buffer(int fd, size_t size) { return nullptr; }

using namespace xinfer;

int main() {
    // Check if we are on Rockchip
    Target target = Target::ROCKCHIP_RKNN;

    // 1. Setup Engine
    auto engine = backends::BackendFactory::create(target);
    // Tell RKNN backend we will use Zero-Copy
    xinfer::Config config;
    config.model_path = "yolo_4k.rknn";
    config.vendor_params = { "ZERO_COPY=TRUE" };
    engine->load_model(config.model_path);

    // 2. Setup RGA Preprocessor
    auto preproc = preproc::create_image_preprocessor(target); // Returns RgaImagePreprocessor
    preproc->init({640, 640, preproc::ImageFormat::RGB});

    // 3. Allocate Special Zero-Copy Tensor
    // Instead of malloc, we allocate a DMA Buffer
    size_t tensor_size = 640*640*3;
    int dma_fd = alloc_dma_buffer(tensor_size);

    // Create Tensor wrapping this FD
    // Note: Core Tensor class needs a constructor for this or set_external_handle
    core::Tensor input_tensor;
    input_tensor.set_external_handle((void*)(intptr_t)dma_fd, tensor_size, core::MemoryType::CmaContiguous);

    // 4. Process Loop
    while (true) {
        // Assume we get a camera frame also as a DMA FD (from V4L2)
        int cam_fd = 100; // Mock

        preproc::ImageFrame frame;
        frame.data = (void*)(intptr_t)cam_fd; // Pass FD instead of pointer
        frame.width = 3840;
        frame.height = 2160;
        frame.format = preproc::ImageFormat::NV12; // Standard Camera Format
        frame.is_device_ptr = true; // Signal it's a handle, not RAM

        // RGA Hardware Resize: 4K NV12 (FD) -> 640p RGB (FD)
        // CPU Usage: 0%
        preproc->process(frame, input_tensor);

        // NPU Inference: Reads directly from FD
        // CPU Usage: 0%
        core::Tensor output;
        engine->predict({input_tensor}, {output});

        // Only now do we touch CPU for post-processing results
        // ...
    }
}