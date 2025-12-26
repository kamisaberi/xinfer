#include <xinfer/zoo/logistics/fill_estimator.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <numeric>

namespace xinfer::zoo::logistics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FillEstimator::Impl {
    FillConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const FillConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("FillEstimator: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Segmentation Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        // Keep analysis at model resolution for speed
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);
    }

    // --- Core Logic ---
    FillResult analyze_mask(const cv::Mat& mask) {
        FillResult result;

        // Count pixels for Cargo vs Void
        int cargo_pixels = 0;
        int void_pixels = 0;
        int total_pixels = mask.rows * mask.cols;

        // Fast iteration
        if (mask.isContinuous()) {
            const uint8_t* ptr = mask.ptr<uint8_t>(0);
            for (int i = 0; i < total_pixels; ++i) {
                if (ptr[i] == config_.cargo_class_id) cargo_pixels++;
                else if (ptr[i] == config_.void_class_id) void_pixels++;
            }
        } else {
            // Slower
            for(int y=0; y<mask.rows; ++y)
                for(int x=0; x<mask.cols; ++x)
                    if (mask.at<uint8_t>(y,x) == config_.cargo_class_id) cargo_pixels++;
        }

        result.void_space_pixels = (float)void_pixels;

        // Calculate Fill Percentage
        int relevant_area = cargo_pixels + void_pixels;
        if (relevant_area > 0) {
            result.fill_percentage = (float)cargo_pixels / relevant_area;
        } else {
            result.fill_percentage = 0.0f;
        }

        return result;
    }

    cv::Mat create_visualization(const cv::Mat& mask, const cv::Size& orig_size) {
        cv::Mat color_mask(mask.size(), CV_8UC3);

        for (int y = 0; y < mask.rows; ++y) {
            for (int x = 0; x < mask.cols; ++x) {
                uint8_t id = mask.at<uint8_t>(y, x);
                if (id == config_.cargo_class_id) {
                    color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0); // Green for Cargo
                } else if (id == config_.void_class_id) {
                    color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // Red for Void
                } else {
                    color_mask.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
                }
            }
        }

        cv::resize(color_mask, color_mask, orig_size, 0, 0, cv::INTER_NEAREST);
        return color_mask;
    }
};

// =================================================================================
// Public API
// =================================================================================

FillEstimator::FillEstimator(const FillConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FillEstimator::~FillEstimator() = default;
FillEstimator::FillEstimator(FillEstimator&&) noexcept = default;
FillEstimator& FillEstimator::operator=(FillEstimator&&) noexcept = default;

FillResult FillEstimator::estimate(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("FillEstimator is null.");

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
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert Tensor -> cv::Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Analyze
    FillResult result = pimpl_->analyze_mask(mask_low);

    // 5. Visualize
    result.visualization = pimpl_->create_visualization(mask_low, image.size());

    return result;
}

} // namespace xinfer::zoo::logistics