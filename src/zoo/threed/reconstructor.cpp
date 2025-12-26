#include <xinfer/zoo/threed/reconstructor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h> // Generic postproc not used for geometry math

#include <iostream>
#include <fstream>
#include <cmath>

namespace xinfer::zoo::threed {

// =================================================================================
// Mesh Helper
// =================================================================================

void Mesh::save_ply(const std::string& filename) const {
    std::ofstream out(filename);
    if (!out.is_open()) {
        XINFER_LOG_ERROR("Failed to open file for writing: " + filename);
        return;
    }

    out << "ply\n";
    out << "format ascii 1.0\n";
    out << "element vertex " << vertices.size() << "\n";
    out << "property float x\n";
    out << "property float y\n";
    out << "property float z\n";
    out << "property uchar red\n";
    out << "property uchar green\n";
    out << "property uchar blue\n";
    out << "end_header\n";

    for (const auto& v : vertices) {
        out << v.x << " " << v.y << " " << v.z << " "
            << (int)v.r << " " << (int)v.g << " " << (int)v.b << "\n";
    }
    out.close();
    XINFER_LOG_INFO("Saved mesh to " + filename);
}

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Reconstructor::Impl {
    ReconstructorConfig config_;

    // AI Components (Depth Model)
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_depth;

    Impl(const ReconstructorConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Reconstructor: Failed to load depth model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        // Standard ImageNet normalization for most Depth models
        pre_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}};

        preproc_->init(pre_cfg);
    }

    // --- Back-Projection Logic ---
    Mesh back_project(const core::Tensor& depth_tensor, const cv::Mat& rgb_image) {
        Mesh mesh;

        // 1. Get Depth Map (Float32)
        // Assume output is [1, 1, H, W] or [1, H, W]
        auto shape = depth_tensor.shape();
        int h = 0, w = 0;
        if (shape.size() == 4) { h = shape[2]; w = shape[3]; }
        else if (shape.size() == 3) { h = shape[1]; w = shape[2]; }

        const float* d_ptr = static_cast<const float*>(depth_tensor.data());

        // We need to resize depth map to match input RGB resolution
        // or resize RGB to match depth map. Resizing depth is better for detail.
        cv::Mat depth_map(h, w, CV_32F, const_cast<float*>(d_ptr));
        cv::Mat depth_resized;

        // Resize depth to match the camera intrinsics resolution (usually input image)
        cv::resize(depth_map, depth_resized, rgb_image.size(), 0, 0, cv::INTER_CUBIC);

        int img_w = rgb_image.cols;
        int img_h = rgb_image.rows;

        // 2. Reserve memory
        // Subsampling factor to reduce point cloud density (optional)
        int stride = 2;
        mesh.vertices.reserve((img_w * img_h) / (stride * stride));

        // 3. Loop pixels
        float fx = config_.fx;
        float fy = config_.fy;
        float cx = config_.cx;
        float cy = config_.cy;

        // Adjust intrinsics if image size differs from config assumptions
        // (Assuming config intrinsics matched original image resolution)

        for (int v = 0; v < img_h; v += stride) {
            for (int u = 0; u < img_w; u += stride) {
                float raw_d = depth_resized.at<float>(v, u);

                // Model specific scaling (MiDaS outputs inverse depth, others metric)
                // Here assumes metric-like output or needs inversion.
                // Simple linear scaling for generic implementation:
                float z = raw_d * config_.depth_scale;

                // Thresholding
                if (z < config_.min_depth || z > config_.max_depth) continue;

                // Back-project
                float x = (u - cx) * z / fx;
                float y = (v - cy) * z / fy;

                // Get Color
                cv::Vec3b color = rgb_image.at<cv::Vec3b>(v, u);

                Vertex vert;
                vert.x = x;
                vert.y = y; // Typically Y is down in CV
                vert.z = z; // Z is forward
                vert.r = color[2]; // BGR -> RGB
                vert.g = color[1];
                vert.b = color[0];

                mesh.vertices.push_back(vert);
            }
        }

        return mesh;
    }
};

// =================================================================================
// Public API
// =================================================================================

Reconstructor::Reconstructor(const ReconstructorConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Reconstructor::~Reconstructor() = default;
Reconstructor::Reconstructor(Reconstructor&&) noexcept = default;
Reconstructor& Reconstructor::operator=(Reconstructor&&) noexcept = default;

void Reconstructor::set_intrinsics(float fx, float fy, float cx, float cy) {
    if (pimpl_) {
        pimpl_->config_.fx = fx;
        pimpl_->config_.fy = fy;
        pimpl_->config_.cx = cx;
        pimpl_->config_.cy = cy;
    }
}

Mesh Reconstructor::reconstruct(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("Reconstructor is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference (Depth Estimation)
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_depth});

    // 3. Postprocess (Geometry Generation)
    return pimpl_->back_project(pimpl_->output_depth, image);
}

} // namespace xinfer::zoo::threed