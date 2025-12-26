#include <xinfer/zoo/geospatial/building_segmenter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>

namespace xinfer::zoo::geospatial {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct BuildingSegmenter::Impl {
    BuildingConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const BuildingConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("BuildingSegmenter: Failed to load model " + config_.model_path);
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
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        post_cfg.apply_softmax = false; // ArgMax is sufficient
        postproc_->init(post_cfg);
    }

    // --- Core Logic: Quantification ---
    void analyze_mask(const cv::Mat& mask, BuildingResult& result) {
        // 1. Filter small noise
        // Assuming Class 1 is "Building".
        cv::Mat binary_mask;
        cv::inRange(mask, cv::Scalar(1), cv::Scalar(1), binary_mask);

        // Morphology to remove salt-and-pepper noise
        cv::morphologyEx(binary_mask, binary_mask, cv::MORPH_OPEN,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5,5)));

        // 2. Count Components
        cv::Mat labels, stats, centroids;
        int num_components = cv::connectedComponentsWithStats(binary_mask, labels, stats, centroids, 8);

        // Component 0 is the background
        result.building_count = num_components - 1;
        result.total_building_area_sq_meters = 0.0f;

        cv::Mat filtered_mask = cv::Mat::zeros(mask.size(), CV_8U);

        // 3. Calculate Area
        for (int i = 1; i < num_components; i++) {
            int area = stats.at<int>(i, cv::CC_STAT_AREA);
            if (area >= config_.min_area_pixels) {
                // Add this component to the final mask
                filtered_mask.setTo(255, labels == i);
            }
        }

        result.footprint_mask = filtered_mask;

        float pixel_area = cv::countNonZero(result.footprint_mask);
        result.total_building_area_sq_meters = pixel_area * config_.sq_meters_per_pixel * config_.sq_meters_per_pixel;
    }

    cv::Mat create_visualization(const cv::Mat& mask, const cv::Mat& original_image) {
        // Resize mask to original image size
        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, original_image.size(), 0, 0, cv::INTER_NEAREST);

        // Create color overlay
        cv::Mat color_layer = cv::Mat::zeros(original_image.size(), CV_8UC3);
        color_layer.setTo(cv::Scalar(0, 0, 255), mask_resized); // Red for buildings

        // Blend
        cv::Mat blended;
        cv::addWeighted(original_image, 1.0, color_layer, 0.4, 0.0, blended);
        return blended;
    }
};

// =================================================================================
// Public API
// =================================================================================

BuildingSegmenter::BuildingSegmenter(const BuildingConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

BuildingSegmenter::~BuildingSegmenter() = default;
BuildingSegmenter::BuildingSegmenter(BuildingSegmenter&&) noexcept = default;
BuildingSegmenter& BuildingSegmenter::operator=(BuildingSegmenter&&) noexcept = default;

BuildingResult BuildingSegmenter::segment(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("BuildingSegmenter is null.");

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
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Analysis
    BuildingResult result;
    pimpl_->analyze_mask(low_res_mask, result);

    // 5. Visualization
    result.visualization = pimpl_->create_visualization(result.footprint_mask, image);

    return result;
}

} // namespace xinfer::zoo::geospatial