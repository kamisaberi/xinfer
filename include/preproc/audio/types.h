#pragma once

#include <cstdint>
#include <vector>

namespace xinfer::preproc {

/**
 * @brief Pixel Format
 * Defines the layout of the input image data.
 */
enum class ImageFormat {
    // Standard CPU/OpenCV Formats
    RGB = 0,
    BGR = 1,
    RGBA = 2,
    BGRA = 3,
    GRAY = 4,

    // Hardware Video Formats (Critical for Rockchip RGA / NVIDIA DeepStream)
    NV12 = 5,  // YUV 4:2:0 Semi-Planar (Y plane + interleaved UV plane)
    NV21 = 6,  // YUV 4:2:0 Semi-Planar (Y plane + interleaved VU plane)
    YUV420P = 7 // Planar YUV (Y + U + V separate planes)
};

/**
 * @brief Interpolation Mode for Resizing
 */
enum class ResizeMode {
    NEAREST = 0, // Fastest, blocky (Good for segmentation masks)
    LINEAR = 1,  // Standard bilinear interpolation (Best balance)
    CUBIC = 2,   // High quality, slower
    AREA = 3     // Best for shrinking images (Moire reduction)
};

/**
 * @brief Memory Location
 * Tells the preprocessor where the source data currently lives.
 */
enum class MemoryLocation {
    HOST_RAM = 0,       // Standard malloc/new
    DEVICE_CUDA = 1,    // GPU VRAM (NVIDIA)
    HOST_PINNED = 2,    // Pinned RAM (Faster PCIe transfer)
    MAPPED_DMA = 3      // Zero-Copy Shared Mem (Rockchip/FPGA/Jetson)
};

/**
 * @brief Rectangle for Cropping
 */
struct Rect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

/**
 * @brief Normalization Parameters
 * Used to convert [0, 255] uint8 -> [-1, 1] or [0, 1] float.
 * Formula: dst = (src - mean) / std
 */
struct NormalizeParams {
    std::vector<float> mean = {0.0f, 0.0f, 0.0f};
    std::vector<float> std  = {1.0f, 1.0f, 1.0f};
    float scale_factor = 1.0f / 255.0f; // Global scaler (e.g., 1/255)
};

/**
 * @brief Configuration for Image Preprocessing Request
 */
struct ImagePreprocConfig {
    // Target Dimensions
    int target_width;
    int target_height;

    // Target Format (usually RGB for models)
    ImageFormat target_format = ImageFormat::RGB;

    // Operations
    ResizeMode resize_mode = ResizeMode::LINEAR;
    bool do_normalize = true;
    NormalizeParams norm_params;

    // Optional: Crop before resize?
    bool do_crop = false;
    Rect crop_rect;

    // Layout: True = NCHW (PyTorch/TensorRT), False = NHWC (TFLite/Rockchip)
    bool layout_nchw = true;
};

} // namespace xinfer::preproc