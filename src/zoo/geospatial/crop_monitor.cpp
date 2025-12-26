#include <xinfer/zoo/geospatial/crop_monitor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <numeric>
#include <map>

namespace xinfer::zoo::geospatial {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CropMonitor::Impl {
    CropConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    // We'll use a custom preprocessing loop for multispectral data
    // std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Visualization
    std::vector<cv::Vec3b> color_lut_;

    Impl(const CropConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CropMonitor: Failed to load model " + config_.model_path);
        }

        // 2. Setup Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);

        // 3. Setup Colors (BGR)
        color_lut_ = {
            cv::Vec3b(42, 42, 165), // Brown (Soil)
            cv::Vec3b(0, 255, 0),   // Green (Healthy)
            cv::Vec3b(0, 255, 255), // Yellow (Stressed)
            cv::Vec3b(0, 0, 255)    // Red (Weed)
        };
    }

    // --- Custom Preprocessing: 4-Channel Normalization ---
    void preprocess(const cv::Mat& img) {
        // Resize
        cv::Mat resized;
        cv::resize(img, resized, cv::Size(config_.input_width, config_.input_height));

        // Convert to float
        cv::Mat float_img;
        resized.convertTo(float_img, CV_32F, 1.0/255.0);

        // Split channels
        std::vector<cv::Mat> channels(config_.input_channels);
        cv::split(float_img, channels);

        // Pack into NCHW tensor
        input_tensor.resize({1, (int64_t)config_.input_channels, (int64_t)config_.input_height, (int64_t)config_.input_width}, core::DataType::kFLOAT);

        float* ptr = static_cast<float*>(input_tensor.data());
        size_t plane_size = config_.input_width * config_.input_height;

        for (int c = 0; c < config_.input_channels; ++c) {
            std::memcpy(ptr + (c * plane_size), channels[c].data, plane_size * sizeof(float));
        }
    }

    // --- Post-processing Logic ---
    void analyze_outputs(const cv::Mat& mask, const cv::Mat& original_img, CropResult& result) {
        int total_pixels = mask.rows * mask.cols;
        std::map<int, int> counts;

        for(int y=0; y<mask.rows; ++y)
            for(int x=0; x<mask.cols; ++x)
                counts[mask.at<uint8_t>(y,x)]++;

        // Calculate Percentages
        for (const auto& kv : counts) {
            int id = kv.first;
            float pct = (float)kv.second / total_pixels;
            if (id < config_.class_names.size()) {
                std::string name = config_.class_names[id];
                if (name == "Healthy") result.stats.percent_healthy = pct;
                else if (name == "Stressed") result.stats.percent_stressed = pct;
                else if (name == "Weed") result.stats.percent_weeds = pct;
                else if (name == "Soil") result.stats.percent_soil = pct;
            }
        }

        // Calculate NDVI
        // NDVI = (NIR - Red) / (NIR + Red)
        cv::Mat resized_orig;
        cv::resize(original_img, resized_orig, mask.size());

        std::vector<cv::Mat> orig_channels;
        cv::split(resized_orig, orig_channels);

        cv::Mat nir, red;
        orig_channels[config_.nir_channel_idx].convertTo(nir, CV_32F);
        orig_channels[config_.red_channel_idx].convertTo(red, CV_32F);

        cv::Mat ndvi_map = (nir - red) / (nir + red + 1e-6);

        // Only average NDVI over vegetated areas (Healthy + Stressed)
        cv::Mat veg_mask = (mask == 1) | (mask == 2);

        cv::Scalar mean_ndvi = cv::mean(ndvi_map, veg_mask);
        result.stats.ndvi_mean = (float)mean_ndvi[0];

        // Create Visualization
        cv::Mat color_map(mask.size(), CV_8UC3);
        for(int y=0; y<mask.rows; ++y)
            for(int x=0; x<mask.cols; ++x)
                color_map.at<cv::Vec3b>(y,x) = color_lut_[mask.at<uint8_t>(y,x)];

        cv::resize(color_map, result.health_map, original_img.size(), 0, 0, cv::INTER_NEAREST);
    }
};

// =================================================================================
// Public API
// =================================================================================

CropMonitor::CropMonitor(const CropConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CropMonitor::~CropMonitor() = default;
CropMonitor::CropMonitor(CropMonitor&&) noexcept = default;
CropMonitor& CropMonitor::operator=(CropMonitor&&) noexcept = default;

CropResult CropMonitor::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("CropMonitor is null.");
    if (image.channels() != pimpl_->config_.input_channels) {
        XINFER_LOG_ERROR("Input image must have " + std::to_string(pimpl_->config_.input_channels) + " channels.");
        return {};
    }

    // 1. Preprocess
    pimpl_->preprocess(image);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert tensor to mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Analysis
    CropResult result;
    pimpl_->analyze_outputs(mask_low, image, result);

    return result;
}

} // namespace xinfer::zoo::geospatial