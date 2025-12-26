#include <xinfer/zoo/vision/low_light_enhancer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory not used; custom image reconstruction logic implemented below.

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct LowLightEnhancer::Impl {
    LowLightConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const LowLightConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("LowLightEnhancer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB; // AI models usually train on RGB
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = config_.scale_factor;

        preproc_->init(pre_cfg);
    }

    // --- Custom Post-Processing: Tensor -> Image ---
    // Handles denormalization, clamping, and channel swapping (RGB->BGR)
    cv::Mat tensor_to_image(const core::Tensor& tensor) {
        auto shape = tensor.shape();
        // Expect [1, 3, H, W] (NCHW) or [1, H, W, 3] (NHWC)
        // Most PyTorch based enhancers are NCHW.

        int c = 0, h = 0, w = 0;
        bool is_nchw = true;

        if (shape.size() == 4) {
            if (shape[1] == 3) { // NCHW
                c = 3; h = (int)shape[2]; w = (int)shape[3];
                is_nchw = true;
            } else if (shape[3] == 3) { // NHWC
                h = (int)shape[1]; w = (int)shape[2]; c = 3;
                is_nchw = false;
            }
        }

        if (c != 3) {
            XINFER_LOG_ERROR("LowLight output shape invalid. Expected 3 channels.");
            return cv::Mat();
        }

        const float* data = static_cast<const float*>(tensor.data());
        cv::Mat result(h, w, CV_8UC3);
        uint8_t* ptr = result.data;
        int spatial_size = h * w;

        // Gamma LUT (Look Up Table) calculation if needed
        // Only apply if gamma != 1.0
        bool apply_gamma = (std::abs(config_.post_gamma - 1.0f) > 0.01f);

        for (int i = 0; i < spatial_size; ++i) {
            float r, g, b;

            if (is_nchw) {
                // Planar Read
                r = data[0 * spatial_size + i];
                g = data[1 * spatial_size + i];
                b = data[2 * spatial_size + i];
            } else {
                // Packed Read
                r = data[i * 3 + 0];
                g = data[i * 3 + 1];
                b = data[i * 3 + 2];
            }

            // Clamp 0.0-1.0
            r = std::max(0.0f, std::min(1.0f, r));
            g = std::max(0.0f, std::min(1.0f, g));
            b = std::max(0.0f, std::min(1.0f, b));

            // Optional Gamma Correction: Output = Input^(1/gamma)
            if (apply_gamma) {
                r = std::pow(r, 1.0f / config_.post_gamma);
                g = std::pow(g, 1.0f / config_.post_gamma);
                b = std::pow(b, 1.0f / config_.post_gamma);
            }

            // Scale to 0-255 and write BGR
            ptr[i * 3 + 0] = static_cast<uint8_t>(b * 255.0f);
            ptr[i * 3 + 1] = static_cast<uint8_t>(g * 255.0f);
            ptr[i * 3 + 2] = static_cast<uint8_t>(r * 255.0f);
        }

        return result;
    }
};

// ====================================================