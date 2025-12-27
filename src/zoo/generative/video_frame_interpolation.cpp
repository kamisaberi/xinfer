#include <xinfer/zoo/generative/video_frame_interpolation.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>

#include <iostream>
#include <algorithm>
#include <cmath>
#include <vector>

namespace xinfer::zoo::generative {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FrameInterpolator::Impl {
    InterpolationConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    // RIFE-like models often take a 6-channel input (Frame0_RGB + Frame1_RGB)
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const InterpolationConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("FrameInterpolator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // RIFE expects 0-255 float input, so no normalization
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

    // --- Core Recursive Interpolation Logic ---
    void interpolate_recursive(const cv::Mat& frame0, const cv::Mat& frame1, int depth, std::vector<cv::Mat>& out_frames) {
        if (depth <= 0) return;

        // 1. Prepare 6-Channel Input
        input_tensor.resize({1, 6, (int64_t)config_.input_height, (int64_t)config_.input_width}, core::DataType::kFLOAT);

        core::Tensor t0, t1;

        preproc::ImageFrame f0{frame0.data, frame0.cols, frame0.rows, preproc::ImageFormat::BGR};
        preproc::ImageFrame f1{frame1.data, frame1.cols, frame1.rows, preproc::ImageFormat::BGR};

        preproc_->process(f0, t0);
        preproc_->process(f1, t1);

        // Concatenate
        char* dst = static_cast<char*>(input_tensor.data());
        size_t single_size = t0.size() * sizeof(float);

        std::memcpy(dst, t0.data(), single_size);
        std::memcpy(dst + single_size, t1.data(), single_size);

        // 2. Inference
        engine_->predict({input_tensor}, {output_tensor});

        // 3. Postprocess
        cv::Mat mid_frame = tensor_to_image(output_tensor);

        // Store
        out_frames.push_back(mid_frame);

        // 4. Recurse
        interpolate_recursive(frame0, mid_frame, depth - 1, out_frames);
        interpolate_recursive(mid_frame, frame1, depth - 1, out_frames);
    }
};

// =================================================================================
// Public API
// =================================================================================

FrameInterpolator::FrameInterpolator(const InterpolationConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FrameInterpolator::~FrameInterpolator() = default;
FrameInterpolator::FrameInterpolator(FrameInterpolator&&) noexcept = default;
FrameInterpolator& FrameInterpolator::operator=(FrameInterpolator&&) noexcept = default;

std::vector<cv::Mat> FrameInterpolator::interpolate(const cv::Mat& frame0, const cv::Mat& frame1) {
    if (!pimpl_) throw std::runtime_error("FrameInterpolator is null.");

    std::vector<cv::Mat> all_frames;

    // upscale_factor = 2 -> depth = 1 (1 frame)
    // upscale_factor = 4 -> depth = 2 (1 + 2 = 3 frames)
    // upscale_factor = 8 -> depth = 3 (1 + 2 + 4 = 7 frames)
    int depth = std::log2(pimpl_->config_.upscale_factor);

    pimpl_->interpolate_recursive(frame0, frame1, depth, all_frames);

    // The recursive approach produces an unsorted list.
    // For a production system, a non-recursive queue-based approach is better to preserve order.
    // Here we just return the collection.

    // For a simple 2x, it just returns the one middle frame.
    if (pimpl_->config_.upscale_factor == 2) {
        return all_frames;
    }

    // TODO: Sort frames by timestamp if needed for higher order interpolation.

    return all_frames;
}

} // namespace xinfer::zoo::generative