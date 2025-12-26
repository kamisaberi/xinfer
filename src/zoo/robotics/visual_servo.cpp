#include <xinfer/zoo/robotics/visual_servo.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <cmath>
#include <algorithm>

namespace xinfer::zoo::robotics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct VisualServo::Impl {
    VisualServoConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const VisualServoConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("VisualServo: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Detector Post-processor
        postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = 0.45f;
        postproc_->init(det_cfg);
    }

    // --- Control Logic ---
    ServoCommand compute_control(const std::vector<postproc::BoundingBox>& detections, int img_w, int img_h) {
        ServoCommand cmd = {0.0f, 0.0f, 0.0f, 0.0f, false};

        // 1. Find Best Target
        // Heuristic: Pick the largest box of the correct class
        const postproc::BoundingBox* target = nullptr;
        float max_area = 0.0f;

        for (const auto& det : detections) {
            if (det.class_id == config_.target_class_id) {
                float w = det.x2 - det.x1;
                float h = det.y2 - det.y1;
                float area = w * h;

                if (area > max_area) {
                    max_area = area;
                    target = &det;
                }
            }
        }

        if (!target) {
            return cmd; // No target, stop (velocity 0)
        }

        cmd.target_acquired = true;

        // 2. Calculate Errors (Normalized -1.0 to 1.0)
        float box_cx = (target->x1 + target->x2) / 2.0f;
        float box_cy = (target->y1 + target->y2) / 2.0f;

        // Error X: -1 (Left) to +1 (Right)
        float err_x = (box_cx - (img_w / 2.0f)) / (img_w / 2.0f);

        // Error Y: -1 (Top) to +1 (Bottom)
        float err_y = (box_cy - (img_h / 2.0f)) / (img_h / 2.0f);

        // Error Depth: (TargetArea - CurrentArea) normalized
        float current_ratio = max_area / (img_w * img_h);
        float err_depth = config_.target_area_ratio - current_ratio;

        // 3. Apply Deadband
        if (std::abs(err_x) < config_.deadband) err_x = 0.0f;
        if (std::abs(err_y) < config_.deadband) err_y = 0.0f;
        if (std::abs(err_depth) < (config_.deadband / 2.0f)) err_depth = 0.0f;

        // 4. Proportional Control
        // Mapping depends on robot kinematic configuration.
        // Assuming a standard Drone config:
        //   err_x -> Yaw Rate (Turn to face)
        //   err_y -> Vertical Velocity (Fly up/down)
        //   err_depth -> Forward Velocity (Approach)

        cmd.yaw_rate = err_x * config_.kp_angular;
        cmd.vz       = -err_y * config_.kp_linear; // Inverted (Target up -> Fly up)
        cmd.vx       = err_depth * config_.kp_depth;

        // Clamp values (Safety)
        // cmd.vx = std::max(-1.0f, std::min(1.0f, cmd.vx));

        return cmd;
    }
};

// =================================================================================
// Public API
// =================================================================================

VisualServo::VisualServo(const VisualServoConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

VisualServo::~VisualServo() = default;
VisualServo::VisualServo(VisualServo&&) noexcept = default;
VisualServo& VisualServo::operator=(VisualServo&&) noexcept = default;

void VisualServo::set_target_id(int class_id) {
    if (pimpl_) pimpl_->config_.target_class_id = class_id;
}

ServoCommand VisualServo::update(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("VisualServo is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect
    auto raw_detections = pimpl_->postproc_->process({pimpl_->output_tensor});

    // 4. Control Logic (Scale boxes back first)
    // We need normalized coords for control, but postproc returns model coords (e.g. 640x640)
    // The compute_control logic handles normalization internally based on image size.
    // However, raw_detections are in model_dims (640x640). We must tell compute_control
    // the dimensions the boxes are currently in.

    return pimpl_->compute_control(raw_detections, pimpl_->config_.input_width, pimpl_->config_.input_height);
}

} // namespace xinfer::zoo::robotics