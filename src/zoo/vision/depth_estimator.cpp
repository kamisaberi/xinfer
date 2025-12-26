#include <xinfer/zoo/vision/depth_estimator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Note: Depth post-processing is specific (normalization/colorization),
// so we handle it here using OpenCV rather than a generic factory.

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DepthEstimator::Impl {
    DepthEstimatorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const DepthEstimatorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DepthEstimator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = 1.0f; // Input assumed 0-255

        preproc_->init(pre_cfg);
    }

    // --- Custom Depth Post-Processing ---
    // Converts raw model output tensor -> User friendly images
    DepthResult postprocess(const core::Tensor& raw_output, const cv::Size& original_size) {
        DepthResult result;

        // 1. Wrap Tensor Data
        // Depth models typically output [1, 1, H, W] or just [1, H, W]
        auto shape = raw_output.shape();
        int model_h = 0;
        int model_w = 0;

        if (shape.size() == 4) {
            model_h = (int)shape[2];
            model_w = (int)shape[3];
        } else if (shape.size() == 3) {
            model_h = (int)shape[1];
            model_w = (int)shape[2];
        } else {
            XINFER_LOG_ERROR("Unexpected depth output shape size: " + std::to_string(shape.size()));
            return result;
        }

        // Create wrapper around raw float data (No copy if on CPU)
        // If data is on GPU, tensor.data() might imply a copy depending on backend implementation.
        // We assume we have host access here.
        const float* ptr = static_cast<const float*>(raw_output.data());
        cv::Mat raw_map(model_h, model_w, CV_32F, const_cast<float*>(ptr));

        // 2. Resize to Original Image Size
        // We use Cubic or Linear interpolation for depth smoothness
        cv::Mat resized_map;
        cv::resize(raw_map, resized_map, original_size, 0, 0, cv::INTER_CUBIC);

        // Clone to ensure we own the memory (uncouple from Tensor)
        result.depth_raw = resized_map.clone();

        // 3. Normalization (Min-Max) for Visualization
        // MiDaS outputs relative inverse depth (disparity).
        // We normalize to 0-255 to apply a colormap.
        double min_val, max_val;
        cv::minMaxLoc(result.depth_raw, &min_val, &max_val);

        if (max_val - min_val > 1e-6) {
            result.depth_raw.convertTo(result.depth_vis, CV_8U, 255.0 / (max_val - min_val), -min_val * 255.0 / (max_val - min_val));
        } else {
            result.depth_vis = cv::Mat::zeros(original_size, CV_8U);
        }

        // 4. Colorization
        // Apply heatmap (Inferno is standard for depth)
        cv::applyColorMap(result.depth_vis, result.depth_vis, config_.colormap);

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DepthEstimator::DepthEstimator(const DepthEstimatorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DepthEstimator::~DepthEstimator() = default;
DepthEstimator::DepthEstimator(DepthEstimator&&) noexcept = default;
DepthEstimator& DepthEstimator::operator=(DepthEstimator&&) noexcept = default;

DepthResult DepthEstimator::estimate(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DepthEstimator is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    // We pass original size to upscale the depth map
    return pimpl_->postprocess(pimpl_->output_tensor, image.size());
}

} // namespace xinfer::zoo::vision