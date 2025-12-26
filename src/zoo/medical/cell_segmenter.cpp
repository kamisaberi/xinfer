#include <xinfer/zoo/medical/cell_segmenter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/instance_seg_interface.h>

#include <iostream>
#include <numeric>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct CellSegmenter::Impl {
    CellConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IInstanceSegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    // YOLO-Seg outputs: Detection head + Proto head
    core::Tensor output_det;
    core::Tensor output_proto;

    // Visualization colors
    std::vector<cv::Scalar> colors_;

    Impl(const CellConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("CellSegmenter: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Postprocessor
        postproc_ = postproc::create_instance_segmentation(config_.target);
        postproc::InstanceSegConfig post_cfg;
        post_cfg.conf_threshold = config_.conf_threshold;
        post_cfg.nms_threshold = config_.nms_threshold;
        post_cfg.target_width = config_.input_width; // Decode at net res, resize later
        post_cfg.target_height = config_.input_height;
        post_cfg.max_detections = 500; // Cells are dense
        postproc_->init(post_cfg);

        // Generate random colors for overlay
        cv::RNG rng(12345);
        for(int i=0; i<100; i++) {
            colors_.push_back(cv::Scalar(rng.uniform(0,255), rng.uniform(0,255), rng.uniform(0,255)));
        }
    }

    // --- Morphometric Analysis ---
    void analyze_cell(const cv::Mat& mask, const cv::Mat& image_gray, CellObject& cell) {
        // 1. Area
        // Count non-zero pixels
        cell.area_pixels = (float)cv::countNonZero(mask);
        cell.area_microns = cell.area_pixels * (config_.microns_per_pixel * config_.microns_per_pixel);

        // 2. Circularity
        // Find contours to get perimeter
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        if (!contours.empty()) {
            double perimeter = cv::arcLength(contours[0], true);
            double area = cv::contourArea(contours[0]);

            if (perimeter > 0) {
                // Circularity = 4 * PI * Area / Perimeter^2
                cell.circularity = (float)((4.0 * M_PI * area) / (perimeter * perimeter));
            } else {
                cell.circularity = 0.0f;
            }
        }

        // 3. Intensity
        // Mean brightness under the mask
        cv::Scalar mean_val = cv::mean(image_gray, mask);
        cell.mean_intensity = (float)mean_val[0];
    }
};

// =================================================================================
// Public API
// =================================================================================

CellSegmenter::CellSegmenter(const CellConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

CellSegmenter::~CellSegmenter() = default;
CellSegmenter::CellSegmenter(CellSegmenter&&) noexcept = default;
CellSegmenter& CellSegmenter::operator=(CellSegmenter&&) noexcept = default;

CellResult CellSegmenter::segment(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("CellSegmenter is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    // Handle input type (Microscopy often B/W or specific stain color)
    frame.format = (image.channels() == 1) ? preproc::ImageFormat::GRAY : preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_det, pimpl_->output_proto});

    // 3. Postprocess
    auto raw_instances = pimpl_->postproc_->process({pimpl_->output_det, pimpl_->output_proto});

    // 4. Analysis Loop
    CellResult result;
    result.segmentation_overlay = image.clone();
    // Ensure overlay is color for visualization even if input was gray
    if (result.segmentation_overlay.channels() == 1) {
        cv::cvtColor(result.segmentation_overlay, result.segmentation_overlay, cv::COLOR_GRAY2BGR);
    }

    cv::Mat gray_ref;
    if (image.channels() == 3) cv::cvtColor(image, gray_ref, cv::COLOR_BGR2GRAY);
    else gray_ref = image;

    float total_area_microns = 0.0f;
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (size_t i = 0; i < raw_instances.size(); ++i) {
        auto& raw = raw_instances[i];

        // Convert raw mask tensor to cv::Mat
        int h = raw.mask.shape()[1];
        int w = raw.mask.shape()[2];
        const uint8_t* ptr = static_cast<const uint8_t*>(raw.mask.data());
        cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));

        // Resize mask to original image resolution
        cv::Mat mask_real;
        cv::resize(mask_low, mask_real, image.size(), 0, 0, cv::INTER_NEAREST);

        // Filter by size before adding
        float px_area = (float)cv::countNonZero(mask_real);
        if (px_area < pimpl_->config_.min_area_px) continue;

        CellObject cell;
        cell.id = (int)i;
        cell.mask = mask_real; // Store full mask

        // Scale Box
        cell.box = raw.box;
        cell.box.x1 *= scale_x; cell.box.x2 *= scale_x;
        cell.box.y1 *= scale_y; cell.box.y2 *= scale_y;

        // Compute Metrics
        pimpl_->analyze_cell(mask_real, gray_ref, cell);

        total_area_microns += cell.area_microns;
        result.cells.push_back(cell);

        // Draw Overlay
        cv::Scalar color = pimpl_->colors_[i % pimpl_->colors_.size()];

        // Find contours for clean outline
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask_real, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        cv::drawContours(result.segmentation_overlay, contours, -1, color, 1);

        // Or semi-transparent fill
        // cv::Mat color_mask(image.size(), CV_8UC3, color);
        // cv::addWeighted(result.segmentation_overlay, 1.0, color_mask, 0.3, 0.0, result.segmentation_overlay, -1, mask_real);
    }

    result.total_count = result.cells.size();
    if (result.total_count > 0) {
        result.average_size_microns = total_area_microns / result.total_count;
    } else {
        result.average_size_microns = 0.0f;
    }

    return result;
}

} // namespace xinfer::zoo::medical