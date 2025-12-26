#include <xinfer/zoo/vision/face_mesh.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
// No generic postproc for 468-point regression; implemented custom below.

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct FaceMesh::Impl {
    FaceMeshConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_landmarks; // Typically [1, 1404] (468 * 3)
    core::Tensor output_score;     // Typically [1, 1]

    Impl(const FaceMeshConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("FaceMesh: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);

        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true; // Most Mesh models use NCHW (1,3,192,192)

        pre_cfg.norm_params.mean = config_.mean;
        pre_cfg.norm_params.std = config_.std;
        pre_cfg.norm_params.scale_factor = config_.scale_factor;

        preproc_->init(pre_cfg);
    }

    // --- Custom Decoding for Mesh Regression ---
    FaceMeshResult decode(const cv::Size& original_size) {
        FaceMeshResult result;
        result.score = 0.0f;

        // 1. Check Score (if model provides it as separate output)
        // Some models output score in a separate tensor, some imply it.
        // Assuming output_score is populated if model has 2 outputs.
        if (output_score.size() > 0) {
            const float* s_ptr = static_cast<const float*>(output_score.data());
            result.score = s_ptr[0]; // Sigmoid usually applied inside model
            if (result.score < config_.score_threshold) {
                return result; // Empty result if low confidence
            }
        } else {
            result.score = 1.0f; // Assume face is present if user passed a crop
        }

        // 2. Parse Landmarks
        // Expected shape: [1, 1404] or [1, 468, 3]
        // 468 points * (x, y, z)
        const float* raw_data = static_cast<const float*>(output_landmarks.data());
        size_t num_values = output_landmarks.size();
        int num_points = num_values / 3;

        result.landmarks.reserve(num_points);

        for (int i = 0; i < num_points; ++i) {
            // Raw values are often coordinate normalized [0, input_dim] or [0, 1]
            // MediaPipe TFLite/ONNX usually outputs [x, y, z] where x,y are pixel coords in 192x192 space
            // and z is depth.

            float raw_x = raw_data[i * 3 + 0];
            float raw_y = raw_data[i * 3 + 1];
            float raw_z = raw_data[i * 3 + 2];

            // Normalize to [0, 1] first if model outputs absolute model coords
            float norm_x = raw_x / config_.input_width;
            float norm_y = raw_y / config_.input_height;

            // Map to original image size
            MeshPoint pt;
            pt.x = norm_x * original_size.width;
            pt.y = norm_y * original_size.height;
            pt.z = raw_z; // Z is usually relative, scaling it is application specific

            result.landmarks.push_back(pt);
        }

        return result;
    }
};

// =================================================================================
// Public API
// =================================================================================

FaceMesh::FaceMesh(const FaceMeshConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

FaceMesh::~FaceMesh() = default;
FaceMesh::FaceMesh(FaceMesh&&) noexcept = default;
FaceMesh& FaceMesh::operator=(FaceMesh&&) noexcept = default;

FaceMeshResult FaceMesh::estimate(const cv::Mat& face_crop) {
    if (!pimpl_) throw std::runtime_error("FaceMesh is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = face_crop.data;
    frame.width = face_crop.cols;
    frame.height = face_crop.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    // Depending on model, might produce 1 or 2 tensors
    // Standard MediaPipe FaceMesh has (Landmarks, Confidence) or just Landmarks.
    // We try to capture both if available.
    try {
        // Attempt to pass both buffers
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_landmarks, pimpl_->output_score});
    } catch (...) {
        // Fallback if model only has 1 output
        pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_landmarks});
    }

    // 3. Decode
    return pimpl_->decode(face_crop.size());
}

void FaceMesh::draw_mesh(cv::Mat& image, const FaceMeshResult& result) {
    if (result.landmarks.empty()) return;

    // Draw simple points
    for (const auto& pt : result.landmarks) {
        cv::circle(image, cv::Point((int)pt.x, (int)pt.y), 1, cv::Scalar(0, 255, 255), -1);
    }

    // Optional: Draw specific contours (Lips, Eyes)
    // Indices for MediaPipe Face Mesh (0-based)
    // Lips Outer: 61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
    // Just drawing a few key points for visualization context
    if (result.landmarks.size() > 4) {
        cv::Point nose = cv::Point((int)result.landmarks[1].x, (int)result.landmarks[1].y); // Tip of nose
        cv::circle(image, nose, 3, cv::Scalar(0, 0, 255), -1);
    }
}

} // namespace xinfer::zoo::vision