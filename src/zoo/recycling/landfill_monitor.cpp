#include <xinfer/zoo/recycling/landfill_monitor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <numeric>

namespace xinfer::zoo::recycling {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct LandfillMonitor::Impl {
    MonitorConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // ROI Mask (Cached)
    cv::Mat roi_mask_;
    float roi_area_pixels_ = 0.0f;

    // Visualization LUT
    std::vector<cv::Vec3b> color_lut_;

    Impl(const MonitorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("LandfillMonitor: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;

        // Standard ImageNet Norm
        pre_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}};
        preproc_->init(pre_cfg);

        // 3. Setup Segmentation Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        // Resize result to input size for analysis
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);

        // 4. Setup Colors
        if (!config_.class_colors.empty()) {
            for (const auto& c : config_.class_colors) {
                if (c.size() >= 3) color_lut_.push_back(cv::Vec3b(c[2], c[1], c[0])); // RGB->BGR
            }
        }
    }

    // Prepare ROI mask on first run (or if image size changes, though we assume fixed stream)
    void check_roi(const cv::Size& img_size) {
        if (roi_mask_.empty() || roi_mask_.size() != img_size) {
            roi_mask_ = cv::Mat::zeros(img_size, CV_8U);

            if (config_.roi_polygon.empty()) {
                // Default: Whole image is ROI
                roi_mask_.setTo(255);
                roi_area_pixels_ = (float)(img_size.width * img_size.height);
            } else {
                // Draw polygon
                std::vector<std::vector<cv::Point>> pts = { config_.roi_polygon };
                cv::fillPoly(roi_mask_, pts, cv::Scalar(255));
                roi_area_pixels_ = (float)cv::countNonZero(roi_mask_);
            }
        }
    }

    // --- Core Logic: Compute Composition ---
    CompositionStats compute_stats(const cv::Mat& class_mask) {
        CompositionStats stats;
        stats.total_area_px = roi_area_pixels_;

        // Histogram calculation
        // We only care about pixels inside the ROI mask
        // Mask is CV_8U (Class IDs)

        // Counters
        std::map<int, int> counts;

        // Iterate (Optimized)
        // If image is large, use cv::calcHist, but for < 1080p, iteration is okay
        int rows = class_mask.rows;
        int cols = class_mask.cols;

        if (class_mask.isContinuous() && roi_mask_.isContinuous()) {
            cols *= rows;
            rows = 1;
        }

        const uint8_t* p_cls = class_mask.ptr<uint8_t>(0);
        const uint8_t* p_roi = roi_mask_.ptr<uint8_t>(0);

        for (int i = 0; i < rows * cols; ++i) {
            if (p_roi[i] > 0) { // Inside ROI
                counts[p_cls[i]]++;
            }
        }

        float total_waste_pixels = 0.0f;

        // Calculate Percentages
        for (const auto& kv : counts) {
            int class_id = kv.first;
            int count = kv.second;

            // Assuming Class 0 is "Background" or "Empty Ground"
            // If class 0 is waste, adjust logic.
            if (class_id > 0) {
                total_waste_pixels += count;
            }

            std::string name = (class_id < (int)config_.class_names.size()) ?
                               config_.class_names[class_id] : "Class_" + std::to_string(class_id);

            stats.material_percentages[name] = (float)count / roi_area_pixels_;
        }

        stats.fill_level = total_waste_pixels / roi_area_pixels_;
        return stats;
    }

    cv::Mat create_visualization(const cv::Mat& class_mask) {
        cv::Mat color_mask(class_mask.size(), CV_8UC3);
        int rows = class_mask.rows;
        int cols = class_mask.cols;

        if (class_mask.isContinuous() && color_mask.isContinuous()) {
            cols *= rows;
            rows = 1;
        }

        const uint8_t* p_cls = class_mask.ptr<uint8_t>(0);
        cv::Vec3b* p_dst = color_mask.ptr<cv::Vec3b>(0);
        size_t lut_size = color_lut_.size();

        for (int i = 0; i < rows * cols; ++i) {
            uint8_t id = p_cls[i];
            if (id < lut_size) {
                p_dst[i] = color_lut_[id];
            } else {
                p_dst[i] = cv::Vec3b(0, 0, 0);
            }
        }
        return color_mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

LandfillMonitor::LandfillMonitor(const MonitorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

LandfillMonitor::~LandfillMonitor() = default;
LandfillMonitor::LandfillMonitor(LandfillMonitor&&) noexcept = default;
LandfillMonitor& LandfillMonitor::operator=(LandfillMonitor&&) noexcept = default;

MonitorResult LandfillMonitor::analyze(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("LandfillMonitor is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Get Class Mask)
    // Mask is returned at config.input_width resolution
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert tensor to Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Stats Analysis
    pimpl_->check_roi(low_res_mask.size());
    MonitorResult result;
    result.stats = pimpl_->compute_stats(low_res_mask);
    result.alert_capacity = (result.stats.fill_level > pimpl_->config_.capacity_threshold);

    // 5. Visualization (Upscale to original)
    cv::Mat color_mask_low = pimpl_->create_visualization(low_res_mask);
    cv::resize(color_mask_low, result.segmentation_vis, image.size(), 0, 0, cv::INTER_NEAREST);

    // Add alpha blending or ROI outline in the application layer, or do it here
    // Here we return the pure color mask

    return result;
}

} // namespace xinfer::zoo::recycling