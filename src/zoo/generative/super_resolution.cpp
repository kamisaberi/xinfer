#include <xinfer/zoo/generative/super_resolution.h>
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

struct SuperResolution::Impl {
    SuperResolutionConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const SuperResolutionConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("SuperResolution: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        // Preprocessor config is minimal; it's mostly used for normalization and layout.
        // Resizing is handled by the tiling logic.
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        // Real-ESRGAN usually takes raw 0-255 float inputs
        pre_cfg.norm_params.scale_factor = 1.0f;
        preproc_->init(pre_cfg);
    }

    // --- Post-processing: Tensor -> Image ---
    cv::Mat tensor_to_image(const core::Tensor& tensor) {
        // Output: [1, 3, H, W] in range [0, 255]
        auto shape = tensor.shape();
        int h = (int)shape[2];
        int w = (int)shape[3];
        int spatial = h * w;

        const float* data = static_cast<const float*>(tensor.data());
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

    // --- Core Tiling Logic ---
    cv::Mat process_with_tiling(const cv::Mat& lr_image) {
        int input_w = lr_image.cols;
        int input_h = lr_image.rows;

        // Output dimensions
        int output_w = input_w * config_.scale;
        int output_h = input_h * config_.scale;
        cv::Mat output_image = cv::Mat::zeros(output_h, output_w, CV_8UC3);

        int tile_size = config_.tile_size;
        int pad = config_.tile_pad;

        // Iterate over image tiles
        for (int y = 0; y < input_h; y += tile_size) {
            for (int x = 0; x < input_w; x += tile_size) {
                // 1. Extract Tile with Padding
                int x1 = std::max(0, x - pad);
                int y1 = std::max(0, y - pad);
                int x2 = std::min(input_w, x + tile_size + pad);
                int y2 = std::min(input_h, y + tile_size + pad);

                cv::Mat tile = lr_image(cv::Rect(x1, y1, x2 - x1, y2 - y1));

                // 2. Preprocess Tile
                preproc::ImageFrame frame{tile.data, tile.cols, tile.rows, preproc::ImageFormat::BGR};

                // Set dynamic target size for preprocessor
                preproc::ImagePreprocConfig cfg;
                cfg.target_width = tile.cols;
                cfg.target_height = tile.rows;
                preproc_->init(cfg);

                preproc_->process(frame, input_tensor);

                // 3. Inference on Tile
                engine_->predict({input_tensor}, {output_tensor});

                // 4. Postprocess Tile
                cv::Mat sr_tile = tensor_to_image(output_tensor);

                // 5. Stitch into Output Image
                // We need to calculate the ROI to copy, removing the padded borders
                int roi_x_start = (x > 0) ? (pad * config_.scale) : 0;
                int roi_y_start = (y > 0) ? (pad * config_.scale) : 0;

                int roi_width = tile_size * config_.scale;
                int roi_height = tile_size * config_.scale;

                // Clamp to sr_tile boundaries
                if (roi_x_start + roi_width > sr_tile.cols) roi_width = sr_tile.cols - roi_x_start;
                if (roi_y_start + roi_height > sr_tile.rows) roi_height = sr_tile.rows - roi_y_start;

                // Destination ROI in final image
                int dest_x = x * config_.scale;
                int dest_y = y * config_.scale;

                if (dest_x + roi_width > output_w) roi_width = output_w - dest_x;
                if (dest_y + roi_height > output_h) roi_height = output_h - dest_y;

                if(roi_width <= 0 || roi_height <= 0) continue;

                cv::Rect src_roi(roi_x_start, roi_y_start, roi_width, roi_height);
                cv::Rect dst_roi(dest_x, dest_y, roi_width, roi_height);

                sr_tile(src_roi).copyTo(output_image(dst_roi));
            }
        }

        return output_image;
    }
};

// =================================================================================
// Public API
// =================================================================================

SuperResolution::SuperResolution(const SuperResolutionConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SuperResolution::~SuperResolution() = default;
SuperResolution::SuperResolution(SuperResolution&&) noexcept = default;
SuperResolution& SuperResolution::operator=(SuperResolution&&) noexcept = default;

cv::Mat SuperResolution::upscale(const cv::Mat& lr_image) {
    if (!pimpl_) throw std::runtime_error("SuperResolution is null.");

    // Simple path (no tiling)
    bool use_tiling = lr_image.cols > pimpl_->config_.tile_size || lr_image.rows > pimpl_->config_.tile_size;

    if (!use_tiling) {
        // 1. Preprocess
        preproc::ImageFrame frame;
        frame.data = lr_image.data;
        frame.width = lr_image.cols;
        frame.height = lr_image.rows;
        frame.format = preproc::ImageFormat::BGR;

        pimpl_->preproc_->init({lr_image.cols, lr_image.rows});
        pimpl_->preproc_->process(frame, pimpl_->input_tensor);

        // 2. Inference
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

        // 3. Postprocess
        return pimpl_->tensor_to_image(pimpl_->output_tensor);
    } else {
        // Tiling Path
        XINFER_LOG_INFO("Input image is large, using tiling...");
        return pimpl_->process_with_tiling(lr_image);
    }
}

} // namespace xinfer::zoo::generative