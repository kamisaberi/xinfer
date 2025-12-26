#include <xinfer/zoo/space/docking_controller.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// Postproc factory skipped; PnP is geometric logic, handled here.

#include <iostream>
#include <cmath>
#include <chrono>

namespace xinfer::zoo::space {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DockingController::Impl {
    DockingConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_kpts; // [1, NumPoints * 2] (x, y)

    // Camera Matrix for PnP
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_; // Assume zero distortion for simplicity, or load from config

    Impl(const DockingConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DockingController: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Camera Matrix
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
            config_.fx, 0, config_.cx,
            0, config_.fy, config_.cy,
            0, 0, 1);
        dist_coeffs_ = cv::Mat::zeros(4, 1, CV_64F);

        // 4. Validate Geometry
        // Ensure the number of 3D points matches the model output
        // We assume model outputs flat [x1, y1, x2, y2 ...]
        // So output size should be config_.target_geometry.size() * 2
    }

    // --- Core Logic: Solve PnP ---
    DockingResult solve_pose(const core::Tensor& kpts_tensor, const cv::Size& img_size) {
        DockingResult result;
        result.status = DockingStatus::SEARCHING;

        // 1. Parse Keypoints
        const float* ptr = static_cast<const float*>(kpts_tensor.data());
        int num_points = config_.target_geometry.size();

        // Scale factors (Model coords -> Image coords)
        float sx = (float)img_size.width / config_.input_width;
        float sy = (float)img_size.height / config_.input_height;

        std::vector<cv::Point2f> image_points;
        float confidence_sum = 0.0f; // If model outputs confidence, use it. Here assuming simple regression.

        for (int i = 0; i < num_points; ++i) {
            float x = ptr[i * 2 + 0] * sx;
            float y = ptr[i * 2 + 1] * sy;
            image_points.emplace_back(x, y);
            result.keypoints_2d.push_back(cv::Point2f(x, y));
        }

        // 2. Solve PnP
        // Returns rotation vector (rvec) and translation vector (tvec)
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(config_.target_geometry,
                                    image_points,
                                    camera_matrix_,
                                    dist_coeffs_,
                                    rvec, tvec,
                                    false, cv::SOLVEPNP_ITERATIVE);

        if (!success) {
            XINFER_LOG_WARN("PnP Solver failed to converge.");
            return result;
        }

        // 3. Fill Result
        result.pose.tx = (float)tvec.at<double>(0);
        result.pose.ty = (float)tvec.at<double>(1);
        result.pose.tz = (float)tvec.at<double>(2); // Z is distance

        // Convert Rodrigues vector to Euler Angles
        cv::Mat rot_mat;
        cv::Rodrigues(rvec, rot_mat);
        // Custom helper to get Euler angles would go here (omitted for brevity)
        // Placeholder:
        result.pose.roll = (float)rvec.at<double>(0) * 180.0f / CV_PI;
        result.pose.pitch = (float)rvec.at<double>(1) * 180.0f / CV_PI;
        result.pose.yaw = (float)rvec.at<double>(2) * 180.0f / CV_PI;

        // 4. Determine Status
        float dist = result.pose.tz;

        // Alignment score: close to center (x,y=0) and flat facing (rot=0)
        float align_err = std::abs(result.pose.tx) + std::abs(result.pose.ty) +
                          std::abs(result.pose.yaw) + std::abs(result.pose.pitch);

        result.pose.alignment_score = std::max(0.0f, 1.0f - (align_err / 10.0f));

        if (dist < 0.2f && result.pose.alignment_score > 0.9f) {
            result.status = DockingStatus::LOCKED;
        } else if (dist < 2.0f) {
            result.status = DockingStatus::TERMINAL;
        } else if (dist < 20.0f) {
            result.status = DockingStatus::APPROACHING;
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

DockingController::DockingController(const DockingConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DockingController::~DockingController() = default;
DockingController::DockingController(DockingController&&) noexcept = default;
DockingController& DockingController::operator=(DockingController&&) noexcept = default;

DockingResult DockingController::calculate_pose(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DockingController is null.");

    auto start = std::chrono::high_resolution_clock::now();

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_kpts});

    // 3. Postprocess (Geometry / PnP)
    DockingResult res = pimpl_->solve_pose(pimpl_->output_kpts, image.size());

    auto end = std::chrono::high_resolution_clock::now();
    res.inference_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    return res;
}

} // namespace xinfer::zoo::space