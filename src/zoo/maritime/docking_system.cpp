#include <xinfer/zoo/maritime/docking_system.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace xinfer::zoo::maritime {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DockingSystem::Impl {
    DockingSystemConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const DockingSystemConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DockingSystem: Failed to load model " + config_.model_path);
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
        postproc_->init(post_cfg);
    }

    // --- Core Geometric Analysis ---
    DockingResult analyze_mask(const cv::Mat& mask, const cv::Mat& original_image) {
        DockingResult result;
        result.state = {999.0f, 999.0f, 999.0f, false}; // Init to safe/unknown values

        // Assume Class 1 = Pier/Land, Class 0 = Water
        cv::Mat pier_mask;
        cv::inRange(mask, cv::Scalar(1), cv::Scalar(1), pier_mask);

        // Find Edges
        cv::Mat edges;
        cv::Canny(pier_mask, edges, 50, 150);

        // Find Lines
        std::vector<cv::Vec4i> lines;
        cv::HoughLinesP(edges, lines, 1, CV_PI / 180, 80, 30, 10);

        if (lines.empty()) return result;

        // Find the "Main" Pier Line (Longest or Most Horizontal)
        cv::Vec4i main_line = lines[0];
        double max_len = 0;

        for (const auto& l : lines) {
            double len = cv::norm(cv::Point(l[0], l[1]) - cv::Point(l[2], l[3]));
            if (len > max_len) {
                max_len = len;
                main_line = l;
            }
        }

        // --- Calculate State ---
        cv::Point p1(main_line[0], main_line[1]);
        cv::Point p2(main_line[2], main_line[3]);

        // 1. Angle
        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        result.state.angle_to_pier = std::atan2(dy, dx) * 180.0f / CV_PI;

        // 2. Distance
        // Heuristic: Take the average Y of the line as the "contact point"
        float line_y = (float)(p1.y + p2.y) / 2.0f;

        // Simple perspective: Assuming distance is proportional to inverse of Y coord from bottom
        // This is a rough proxy and should be replaced with calibrated geometry.
        float dist_normalized = (float)(mask.rows - line_y) / mask.rows;
        result.state.distance_to_pier = (1.0f - dist_normalized) * 100.0f; // Scale to arbitrary meters

        // 3. Lateral Offset
        // Center of line vs. Center of image
        float line_cx = (float)(p1.x + p2.x) / 2.0f;
        float img_cx = (float)mask.cols / 2.0f;
        result.state.lateral_offset = (line_cx - img_cx) * (config_.mm_per_pixel / 1000.0f);

        // 4. Alignment Check
        result.state.is_aligned = (std::abs(result.state.angle_to_pier) < config_.angle_tolerance_deg &&
                                   result.state.distance_to_pier < config_.distance_tolerance_m);

        // 5. Visualization
        if (original_image.channels() == 1) {
            cv::cvtColor(original_image, result.visualization, cv::COLOR_GRAY2BGR);
        } else {
            result.visualization = original_image.clone();
        }

        // Scale line to original image size
        float sx = (float)original_image.cols / mask.cols;
        float sy = (float)original_image.rows / mask.rows;

        cv::line(result.visualization, cv::Point(p1.x*sx, p1.y*sy), cv::Point(p2.x*sx, p2.y*sy), cv::Scalar(0, 0, 255), 3, cv::LINE_AA);

        std::string text = "Dist: " + std::to_string(result.state.distance_to_pier) + "m Angle: " + std::to_string(result.state.angle_to_pier) + " deg";
        cv::putText(result.visualization, text, cv::Point(20, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,255,0), 2);

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DockingSystem::DockingSystem(const DockingSystemConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DockingSystem::~DockingSystem() = default;
DockingSystem::DockingSystem(DockingSystem&&) noexcept = default;
DockingSystem& DockingSystem::operator=(DockingSystem&&) noexcept = default;

DockingResult DockingSystem::analyze_frame(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DockingSystem is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (ArgMax)
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Convert Tensor -> cv::Mat
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat low_res_mask(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Geometric Analysis
    return pimpl_->analyze_mask(low_res_mask, image);
}

} // namespace xinfer::zoo::maritime