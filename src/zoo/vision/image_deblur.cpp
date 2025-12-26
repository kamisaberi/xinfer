#include <xinfer/zoo/vision/image_deblur.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory is not used here; image restoration requires
// specific tensor-to-image reconstruction logic implemented below.

#include <iostream>
#include <algorithm>
#include <chrono>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct ImageDeblur::Impl {
    ImageDeblurConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const ImageDeblurConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("ImageDeblur: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB; // Models usually trained on RGB
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = config_.scale_factor;

        preproc_->init(pre_cfg);
    }

    // --- Custom Post-Processing: Tensor -> Image ---
    cv::Mat tensor_to_image(const core::Tensor& tensor) {
        auto shape = tensor.shape(); // Expect [1, 3, H, W]
        if (shape.size() != 4) {
            XINFER_LOG_ERROR("Deblur output shape mismatch. Expected 4 dims.");
            return cv::Mat();
        }

        int c = (int)shape[1];
        int h = (int)shape[2];
        int w = (int)shape[3];
        int spatial_size = h * w;

        // Pointer to float data (on CPU)
        // If data is on GPU, we assume core::Tensor handles sync/mapping access
        const float* data = static_cast<const float*>(tensor.data());

        // Create output Mat
        cv::Mat result(h, w, CV_8UC3);
        uint8_t* ptr = result.data;

        // Perform NCHW (Planar) -> HWC (Packed) conversion + Denormalization
        // Loop over pixels
        for (int i = 0; i < spatial_size; ++i) {
            // Read R, G, B planes
            float r = data[0 * spatial_size + i];
            float g = data[1 * spatial_size + i];
            float b = data[2 * spatial_size + i];

            // Clamp 0.0-1.0 -> 0-255
            // Note: If model output is -1 to 1, math changes here.
            // Assuming standard [0,1] output for restoration.
            int ir = std::min(std::max(int(r * 255.0f), 0), 255);
            int ig = std::min(std::max(int(g * 255.0f), 0), 255);
            int ib = std::min(std::max(int(b * 255.0f), 0), 255);

            // Write BGR (OpenCV standard)
            ptr[i * 3 + 0] = (uint8_t)ib;
            ptr[i * 3 + 1] = (uint8_t)ig;
            ptr[i * 3 + 2] = (uint8_t)ir;
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

ImageDeblur::ImageDeblur(const ImageDeblurConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

ImageDeblur::~ImageDeblur() = default;
ImageDeblur::ImageDeblur(ImageDeblur&&) noexcept = default;
ImageDeblur& ImageDeblur::operator=(ImageDeblur&&) noexcept = default;

DeblurResult ImageDeblur::process(const cv::Mat& blurry_image) {
    if (!pimpl_) throw std::runtime_error("ImageDeblur is null.");

    auto start = std::chrono::high_resolution_clock::now();

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = blurry_image.data;
    frame.width = blurry_image.cols;
    frame.height = blurry_image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    auto end = std::chrono::high_resolution_clock::now();
    float time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    // 3. Postprocess (Reconstruct Image)
    cv::Mat sharp = pimpl_->tensor_to_image(pimpl_->output_tensor);

    // Optional: If model output size differs from input, resize back
    if (sharp.cols != blurry_image.cols || sharp.rows != blurry_image.rows) {
        cv::resize(sharp, sharp, blurry_image.size());
    }

    return {sharp, time_ms};
}

} // namespace xinfer::zoo::vision