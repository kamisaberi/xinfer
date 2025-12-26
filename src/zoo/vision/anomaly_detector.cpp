#include <xinfer/zoo/vision/anomaly_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct AnomalyDetector::Impl {
    AnomalyConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IAnomalyPostprocessor> postproc_;

    // Data Containers
    // We need both input and output tensors to compute the difference
    core::Tensor input_tensor;
    core::Tensor output_tensor; // The reconstructed image

    Impl(const AnomalyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        // Optional: Pass params like "ZERO_COPY=TRUE" for Rockchip if needed

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("AnomalyDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // Set Normalization (Crucial for AE models)
        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = config_.scale;

        preproc_->init(pre_cfg);

        // 3. Setup Postprocessor (Diff Calculator)
        // If Target=NVIDIA, this uses CUDA kernels to compute MSE.
        // If Target=CPU/Other, this uses OpenCV AVX.
        postproc_ = postproc::create_anomaly(config_.target);

        postproc::AnomalyConfig post_cfg;
        post_cfg.threshold = config_.threshold;
        post_cfg.use_smoothing = config_.use_smoothing;
        postproc_->init(post_cfg);
    }

    // Helper: Convert Float Tensor to CV::Mat Heatmap
    cv::Mat tensor_to_heatmap(const core::Tensor& t) {
        // Tensor is [1, 1, H, W] Float32 in range [0, 1] usually (or higher errors)
        auto shape = t.shape();
        int h = (int)shape[2];
        int w = (int)shape[3];

        // Copy to CPU Mat
        cv::Mat float_map(h, w, CV_32F);
        // Note: tensor.copy_to_host handles device-to-host if needed
        // Assuming tensor.data() returns host accessible pointer for CPU postproc results
        // or we use a copy method. Here assuming host pointer.
        std::memcpy(float_map.data, t.data(), h * w * sizeof(float));

        // Normalize for display (0-255)
        // We assume errors > 1.0 are max saturated
        cv::Mat uint_map;
        float_map.convertTo(uint_map, CV_8U, 255.0);

        // Apply Jet Colormap
        cv::Mat color_map;
        cv::applyColorMap(uint_map, color_map, cv::COLORMAP_JET);
        return color_map;
    }

    cv::Mat tensor_to_mask(const core::Tensor& t) {
        // Tensor is [1, 1, H, W] Uint8 (0 or 255)
        auto shape = t.shape();
        int h = (int)shape[2];
        int w = (int)shape[3];

        cv::Mat mask(h, w, CV_8U);
        std::memcpy(mask.data, t.data(), h * w * sizeof(uint8_t));
        return mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

AnomalyDetector::AnomalyDetector(const AnomalyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

AnomalyDetector::~AnomalyDetector() = default;
AnomalyDetector::AnomalyDetector(AnomalyDetector&&) noexcept = default;
AnomalyDetector& AnomalyDetector::operator=(AnomalyDetector&&) noexcept = default;

AnomalyResult AnomalyDetector::inspect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("AnomalyDetector is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    // input_tensor holds the Normalized, Resized Float32 image
    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference (Reconstruction)
    // Autoencoder takes input, returns reconstructed version
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Compare)
    // Calculates |Input - Output|, reduces channels, smoothes, thresholds.
    auto raw_res = pimpl_->postproc_->process(pimpl_->input_tensor, pimpl_->output_tensor);

    // 4. Format Output
    AnomalyResult res;
    res.is_anomaly = raw_res.is_anomaly;
    res.score = raw_res.anomaly_score;

    // Convert internal tensors to OpenCV Mats for easy visualization
    res.heatmap = pimpl_->tensor_to_heatmap(raw_res.heatmap);

    if (res.is_anomaly) {
        res.segmentation = pimpl_->tensor_to_mask(raw_res.segmentation_mask);
    } else {
        // Empty black mask if no anomaly
        res.segmentation = cv::Mat::zeros(pimpl_->config_.input_height, pimpl_->config_.input_width, CV_8U);
    }

    return res;
}

} // namespace xinfer::zoo::vision