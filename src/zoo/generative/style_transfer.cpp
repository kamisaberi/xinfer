#include <xinfer/zoo/generative/style_transfer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct StyleTransfer::Impl {
    StyleTransferConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const StyleTransferConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("StyleTransfer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // Style models usually don't use standard normalization,
        // they just scale 0-255 to 0-1 or -1 to 1.
        // For this, we assume the model takes raw 0-255 RGB, so no scaling.
        pre_cfg.norm_params.scale_factor = 1.0f;
        pre_cfg.norm_params.mean = {0,0,0};
        pre_cfg.norm_params.std = {1,1,1};

        preproc_->init(pre_cfg);
    }

    // --- Custom Post-processing: Tensor -> Image ---
    cv::Mat tensor_to_image() {
        // Output is [1, 3, H, W] in range [0, 255] (usually)
        auto shape = output_tensor.shape();
        int c = (int)shape[1];
        int h = (int)shape[2];
        int w = (int)shape[3];
        int spatial = h * w;

        const float* data = static_cast<const float*>(output_tensor.data());
        cv::Mat result(h, w, CV_8UC3);
        uint8_t* ptr = result.data;

        // NCHW -> HWC + Clamp
        for (int i = 0; i < spatial; ++i) {
            float r = data[0 * spatial + i];
            float g = data[1 * spatial + i];
            float b = data[2 * spatial + i];

            ptr[i * 3 + 0] = (uint8_t)std::clamp(b, 0.0f, 255.0f);
            ptr[i * 3 + 1] = (uint8_t)std::clamp(g, 0.0f, 255.0f);
            ptr[i * 3 + 2] = (uint8_t)std::clamp(r, 0.0f, 255.0f);
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

StyleTransfer::StyleTransfer(const StyleTransferConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

StyleTransfer::~StyleTransfer() = default;
StyleTransfer::StyleTransfer(StyleTransfer&&) noexcept = default;
StyleTransfer& StyleTransfer::operator=(StyleTransfer&&) noexcept = default;

cv::Mat StyleTransfer::apply(const cv::Mat& content_image) {
    if (!pimpl_) throw std::runtime_error("StyleTransfer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = content_image.data;
    frame.width = content_image.cols;
    frame.height = content_image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    cv::Mat styled = pimpl_->tensor_to_image();

    // Resize to original content size
    if (styled.size() != content_image.size()) {
        cv::resize(styled, styled, content_image.size(), 0, 0, cv::INTER_CUBIC);
    }

    return styled;
}

} // namespace xinfer::zoo::generative