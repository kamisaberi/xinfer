#include <xinfer/zoo/medical/ultrasound_guide.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/segmentation_interface.h>

#include <iostream>
#include <cmath>
#include <vector>

namespace xinfer::zoo::medical {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct UltrasoundGuide::Impl {
    UltrasoundConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Visualization Colors (BGR)
    const cv::Scalar COL_TARGET = {0, 255, 0};    // Green
    const cv::Scalar COL_DANGER = {0, 0, 255};    // Red
    const cv::Scalar COL_NEEDLE = {255, 255, 0};  // Cyan/Yellow
    const cv::Scalar COL_TRAJ   = {255, 0, 255};  // Magenta

    Impl(const UltrasoundConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("UltrasoundGuide: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::GRAY; // Ultrasound is mono
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Segmentation Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        // We output at model resolution (usually sufficient) or upscale
        // Upscaling to input size is safer for precise measurements
        post_cfg.target_width = config_.input_width;
        post_cfg.target_height = config_.input_height;
        postproc_->init(post_cfg);
    }

    // --- Geometric Analysis ---
    void analyze_scene(const cv::Mat& mask, const cv::Size& orig_size, GuideResult& result) {
        // Mask is at config_.input_size, likely different from orig_size.
        // We will do math in model space and scale results for display if needed.
        // For simplicity, we assume we visualize on a resized image or the mask matches orig.
        // Let's assume mask is resized to orig_size by postproc if we configured it so?
        // Actually postproc resizes to config_.input_width.
        // We should map back to orig_size for visualization.

        float scale_x = (float)orig_size.width / mask.cols;
        float scale_y = (float)orig_size.height / mask.rows;

        // 1. Extract Masks for each class
        cv::Mat mask_target = (mask == config_.class_id_target);
        cv::Mat mask_danger = (mask == config_.class_id_danger);
        cv::Mat mask_needle = (mask == config_.class_id_needle);

        // 2. Find Target Centroid
        cv::Moments m_tgt = cv::moments(mask_target, true);
        cv::Point2f pt_target;
        if (m_tgt.m00 > 0) {
            pt_target = cv::Point2f(m_tgt.m10 / m_tgt.m00, m_tgt.m01 / m_tgt.m00);
            result.target_visible = true;
        }

        // 3. Fit Needle Line
        std::vector<cv::Point> needle_pts;
        cv::findNonZero(mask_needle, needle_pts);

        if (needle_pts.size() > 10) { // Threshold to ignore noise
            result.needle_visible = true;

            // Fit Line (vx, vy, x0, y0)
            cv::Vec4f line;
            cv::fitLine(needle_pts, line, cv::DIST_L2, 0, 0.01, 0.01);

            cv::Point2f dir(line[0], line[1]);
            cv::Point2f origin(line[2], line[3]);

            // Determine Tip (Point on line closest to target, or extreme point of mask)
            // Heuristic: Find the point in needle_pts that is furthest in the direction of the target
            // Simple approach: The point in needle_pts closest to pt_target

            cv::Point2f tip = origin;
            float min_dist = 1e9;

            if (result.target_visible) {
                // If target known, find tip closest to target
                for(auto& p : needle_pts) {
                    float d = cv::norm(cv::Point2f(p) - pt_target);
                    if (d < min_dist) { min_dist = d; tip = p; }
                }
                result.distance_to_target_mm = min_dist * config_.mm_per_pixel;
            } else {
                // Just find 'deepest' point (max Y usually in ultrasound)
                float max_y = -1.0f;
                for(auto& p : needle_pts) {
                    if (p.y > max_y) { max_y = p.y; tip = p; }
                }
            }

            // --- Projection & Collision Check ---
            // Project line from tip by 5cm (converted to pixels)
            float project_len = 50.0f / config_.mm_per_pixel;

            // Determine direction vector (ensure it points down/towards target)
            // If dot(dir, target-origin) < 0, flip dir
            if (result.target_visible) {
                cv::Point2f vec_to_tgt = pt_target - origin;
                if (vec_to_tgt.dot(dir) < 0) dir = -dir;
            } else {
                if (dir.y < 0) dir = -dir; // Assume needle goes down
            }

            cv::Point2f end_pt = tip + dir * project_len;

            // Draw Trajectory on Overlay
            // Scale points to original image size
            cv::Point2f tip_real(tip.x * scale_x, tip.y * scale_y);
            cv::Point2f end_real(end_pt.x * scale_x, end_pt.y * scale_y);

            cv::line(result.overlay, tip_real, end_real, COL_TRAJ, 2, cv::LINE_AA);
            cv::circle(result.overlay, tip_real, 4, COL_NEEDLE, -1);

            // Check collision with Danger Mask along the line
            // We sample the line in the mask space
            cv::LineIterator it(mask_danger, tip, end_pt, 8);
            for(int i = 0; i < it.count; i++, ++it) {
                if (mask_danger.at<uint8_t>(it.pos())) {
                    result.warning_collision = true;
                    cv::putText(result.overlay, "WARNING: ARTERY", cv::Point(20, 50),
                                cv::FONT_HERSHEY_SIMPLEX, 1.0, COL_DANGER, 3);
                    break;
                }
            }
        }
    }

    void draw_overlays(const cv::Mat& mask, const cv::Size& orig_size, cv::Mat& overlay) {
        // Resize mask to output size
        cv::Mat mask_resized;
        cv::resize(mask, mask_resized, orig_size, 0, 0, cv::INTER_NEAREST);

        // Create color layer
        cv::Mat color_layer = cv::Mat::zeros(orig_size, CV_8UC3);

        // Fast color mapping using LUT or logic
        // Using pointer iteration for speed
        int rows = color_layer.rows;
        int cols = color_layer.cols;
        if (mask_resized.isContinuous() && color_layer.isContinuous()) {
            cols *= rows; rows = 1;
        }

        const uint8_t* p_mask = mask_resized.ptr<uint8_t>(0);
        cv::Vec3b* p_dst = color_layer.ptr<cv::Vec3b>(0);

        for (int i = 0; i < rows * cols; ++i) {
            uint8_t id = p_mask[i];
            if (id == config_.class_id_target) {
                p_dst[i] = cv::Vec3b(COL_TARGET[0], COL_TARGET[1], COL_TARGET[2]);
            } else if (id == config_.class_id_danger) {
                p_dst[i] = cv::Vec3b(COL_DANGER[0], COL_DANGER[1], COL_DANGER[2]);
            } else if (id == config_.class_id_needle) {
                p_dst[i] = cv::Vec3b(COL_NEEDLE[0], COL_NEEDLE[1], COL_NEEDLE[2]);
            }
        }

        // Alpha Blend
        cv::addWeighted(overlay, 1.0, color_layer, config_.alpha_blend, 0, overlay);
    }
};

// =================================================================================
// Public API
// =================================================================================

UltrasoundGuide::UltrasoundGuide(const UltrasoundConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

UltrasoundGuide::~UltrasoundGuide() = default;
UltrasoundGuide::UltrasoundGuide(UltrasoundGuide&&) noexcept = default;
UltrasoundGuide& UltrasoundGuide::operator=(UltrasoundGuide&&) noexcept = default;

GuideResult UltrasoundGuide::process(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("UltrasoundGuide is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = (image.channels() == 1) ? preproc::ImageFormat::GRAY : preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess (Get Class Mask)
    // Mask returned at model resolution
    auto seg_res = pimpl_->postproc_->process(pimpl_->output_tensor);

    // Wrap Tensor
    int h = seg_res.mask.shape()[1];
    int w = seg_res.mask.shape()[2];
    const uint8_t* ptr = static_cast<const uint8_t*>(seg_res.mask.data());
    cv::Mat mask_low(h, w, CV_8U, const_cast<uint8_t*>(ptr));

    // 4. Analysis & Viz
    GuideResult result;
    result.target_visible = false;
    result.needle_visible = false;
    result.warning_collision = false;
    result.distance_to_target_mm = -1.0f;

    // Prepare overlay canvas (convert input to BGR if needed)
    if (image.channels() == 1) {
        cv::cvtColor(image, result.overlay, cv::COLOR_GRAY2BGR);
    } else {
        result.overlay = image.clone();
    }

    // Draw Segmentation
    pimpl_->draw_overlays(mask_low, image.size(), result.overlay);

    // Calc Geometry & Draw Lines
    pimpl_->analyze_scene(mask_low, image.size(), result);

    return result;
}

} // namespace xinfer::zoo::medical