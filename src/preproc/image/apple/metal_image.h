#pragma once

#include <xinfer/preproc/image/image_preprocessor.h>
#include <xinfer/preproc/image/types.h>
#include <memory>

namespace xinfer::preproc {

/**
 * @brief Apple Metal Image Preprocessor
 * 
 * Optimized for Apple Silicon (M1/M2/M3) and iOS devices.
 * Uses Metal Performance Shaders (MPS) for resizing and 
 * Custom Compute Kernels for Normalization/Layout conversion.
 * 
 * Features:
 * - Unified Memory support (Zero-copy potential).
 * - Hardware accelerated Bilinear/Lanczos resizing.
 */
class MetalImagePreprocessor : public IImagePreprocessor {
public:
    MetalImagePreprocessor();
    ~MetalImagePreprocessor() override;

    void init(const ImagePreprocConfig& config) override;

    /**
     * @brief Process image on Apple GPU.
     * 
     * 1. Uploads CPU bytes to MTLTexture (fast via Unified Memory).
     * 2. Resizes via MPSImageScale.
     * 3. Runs 'preprocess_nchw' compute kernel.
     * 4. syncs/copies back to Tensor memory.
     */
    void process(const ImageFrame& src, core::Tensor& dst) override;

private:
    struct Impl; // Hide Objective-C++ implementation details
    std::unique_ptr<Impl> m_impl;

    ImagePreprocConfig m_config;
};

} // namespace xinfer::preproc