#include <xinfer/zoo/hci/gaze_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

// We reuse the standard detector for landmarks
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <cmath>
#include <algorithm>

namespace xinfer::zoo::hci {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct GazeTracker::Impl {
    GazeConfig config_;

    // --- Components ---
    std::unique_ptr<vision::ObjectDetector> landmark_detector_; // Simplified: treat landmarks as objects
    std::unique_ptr<backends::IBackend> gaze_engine_;

    // Preprocessors for different inputs
    std::unique_ptr<preproc::IImagePreprocessor> face_preproc_;
    std::unique_ptr<preproc::IImagePreprocessor> eye_preproc_;

    // --- Tensors ---
    core::Tensor face_tensor, left_eye_tensor, right_eye_tensor, head_pose_tensor;
    core::Tensor output_gaze_tensor;

    Impl(const GazeConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Landmark Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.landmark_detector_path;
        // In a real app, this would be a specialized landmark detector
        // Using ObjectDetector as a stand-in
        landmark_detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);

        // 2. Setup Gaze Engine
        gaze_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config gaze_cfg; gaze_cfg.model_path = config_.gaze_model_path;
        if (!gaze_engine_->load_model(gaze_cfg.model_path)) {
            throw std::runtime_error("GazeTracker: Failed to load gaze model.");
        }

        // 3. Setup Preprocessors
        // For Face
        face_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig fp_cfg;
        fp_cfg.target_width = config_.face_input_size;
        fp_cfg.target_height = config_.face_input_size;
        face_preproc_->init(fp_cfg);

        // For Eyes
        eye_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig ep_cfg;
        ep_cfg.target_width = config_.eye_input_size;
        ep_cfg.target_height = config_.eye_input_size;
        eye_preproc_->init(ep_cfg);
    }

    // --- Core Logic ---
    GazeResult process_gaze(const cv::Mat& image) {
        GazeResult result;
        result.confidence = 0.0f;

        // 1. Detect Face & Landmarks (Using Detector as proxy)
        auto detections = landmark_detector_->predict(image);
        if (detections.empty()) return result; // No face

        // Find best (largest) face
        auto best_face = std::max_element(detections.begin(), detections.end(),
            [](const auto& a, const auto& b) {
                return (a.x2-a.x1) * (a.y2-a.y1) < (b.x2-b.x1) * (b.y2-b.y1);
            });

        // In a real app, 'detections' would contain landmarks.
        // We simulate some landmark points based on the face box.
        cv::Rect face_box(best_face->x1, best_face->y1, best_face->x2-best_face->x1, best_face->y2-best_face->y1);
        cv::Point left_eye_center(face_box.x + face_box.width * 0.7, face_box.y + face_box.height * 0.4);
        cv::Point right_eye_center(face_box.x + face_box.width * 0.3, face_box.y + face_box.height * 0.4);

        // 2. Estimate Head Pose (Simple PnP)
        // ... (Logic to solvePnP from landmarks -> rvec, tvec) ...
        // We create a dummy pose tensor for now
        head_pose_tensor.resize({1, 2}, core::DataType::kFLOAT); // Pitch, Yaw
        float* p_ptr = (float*)head_pose_tensor.data();
        p_ptr[0] = 0.0f; p_ptr[1] = 0.0f; // Assume facing forward

        // 3. Crop & Preprocess Inputs
        cv::Mat face_crop = image(face_box);

        // Eye crops
        float eye_w = face_box.width * 0.3;
        cv::Rect left_eye_roi(left_eye_center.x - eye_w/2, left_eye_center.y - eye_w/2, eye_w, eye_w);
        cv::Rect right_eye_roi(right_eye_center.x - eye_w/2, right_eye_center.y - eye_w/2, eye_w, eye_w);

        cv::Mat left_crop = image(left_eye_roi & cv::Rect(0,0,image.cols, image.rows));
        cv::Mat right_crop = image(right_eye_roi & cv::Rect(0,0,image.cols, image.rows));

        // Process with correct preprocessor
        face_preproc_->process({face_crop.data, face_crop.cols, face_crop.rows, preproc::ImageFormat::BGR}, face_tensor);
        eye_preproc_->process({left_crop.data, left_crop.cols, left_crop.rows, preproc::ImageFormat::BGR}, left_eye_tensor);
        eye_preproc_->process({right_crop.data, right_crop.cols, right_crop.rows, preproc::ImageFormat::BGR}, right_eye_tensor);

        // 4. Inference
        // Model takes multiple inputs
        gaze_engine_->predict({face_tensor, left_eye_tensor, right_eye_tensor, head_pose_tensor}, {output_gaze_tensor});

        // 5. Decode
        // Output is typically [Pitch, Yaw] in radians.
        const float* gaze_ptr = static_cast<const float*>(output_gaze_tensor.data());
        float pitch = gaze_ptr[0];
        float yaw = gaze_ptr[1];

        // Convert to Cartesian Vector
        result.gaze_vector.x = -std::cos(pitch) * std::sin(yaw);
        result.gaze_vector.y = -std::sin(pitch);
        result.gaze_vector.z = -std::cos(pitch) * std::cos(yaw);
        result.confidence = 1.0f; // Placeholder

        // 6. Calculate Point of Regard (PoR)
        // Find intersection of gaze ray with screen plane
        // Ray: P = EyeOrigin + t * GazeVector
        // Plane: Z = screen_distance

        // Assume eye origin is roughly center of face box
        // (In real app, use 3D landmarks + pose)
        float t = config_.screen_distance_m / result.gaze_vector.z;
        if (t > 0) {
            float x_on_screen = t * result.gaze_vector.x;
            float y_on_screen = t * result.gaze_vector.y;

            // Convert meters -> pixels
            result.point_of_regard.x = (x_on_screen / config_.screen_width_m + 0.5f) * image.cols;
            result.point_of_regard.y = (y_on_screen / config_.screen_height_m + 0.5f) * image.rows;
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

GazeTracker::GazeTracker(const GazeConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GazeTracker::~GazeTracker() = default;
GazeTracker::GazeTracker(GazeTracker&&) noexcept = default;
GazeTracker& GazeTracker::operator=(GazeTracker&&) noexcept = default;

GazeResult GazeTracker::track(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("GazeTracker is null.");

    return pimpl_->process_gaze(image);
}

} // namespace xinfer::zoo::hci