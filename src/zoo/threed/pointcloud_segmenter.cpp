#include <xinfer/zoo/threed/pointcloud_segmenter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/postproc/factory.h>
// Note: Preprocessing is highly specific (3D->2D projection), implemented here.

#include <iostream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstring>

namespace xinfer::zoo::threed {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PointCloudSegmenter::Impl {
    PointCloudSegConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // We reuse the 2D Segmentation Post-processor for the Range Image
    std::unique_ptr<postproc::ISegmentationPostprocessor> postproc_;

    // Data Containers
    core::Tensor input_tensor; // [1, 5, H, W]
    core::Tensor output_tensor; // [1, NumClasses, H, W]

    // Projection Map: Stores which 3D point maps to which 2D pixel
    // Used to map predictions back to 3D.
    // Index = y * w + x; Value = index in input 'points' vector
    std::vector<int> proj_idx_map_;

    Impl(const PointCloudSegConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PointCloudSegmenter: Failed to load model " + config_.model_path);
        }

        // 2. Pre-allocate Input Tensor
        // Channels: 5 (Range, X, Y, Z, Intensity)
        input_tensor.resize({1, 5, (int64_t)config_.proj_height, (int64_t)config_.proj_width}, core::DataType::kFLOAT);

        // Map buffer size
        proj_idx_map_.resize(config_.proj_height * config_.proj_width);

        // 3. Setup Post-processor
        postproc_ = postproc::create_segmentation(config_.target);
        postproc::SegmentationConfig post_cfg;
        post_cfg.target_width = config_.proj_width;
        post_cfg.target_height = config_.proj_height;
        post_cfg.apply_softmax = false; // ArgMax directly
        postproc_->init(post_cfg);
    }

    // --- Spherical Projection (3D -> 2D) ---
    void project_points(const std::vector<PointXYZI>& points) {
        // Reset buffers
        float* data = static_cast<float*>(input_tensor.data());
        std::memset(data, 0, input_tensor.size() * sizeof(float));
        std::fill(proj_idx_map_.begin(), proj_idx_map_.end(), -1);

        int H = config_.proj_height;
        int W = config_.proj_width;
        float fov_up = config_.fov_up * M_PI / 180.0f;
        float fov_down = config_.fov_down * M_PI / 180.0f;
        float fov = std::abs(fov_up - fov_down);

        int plane_size = H * W;

        for (size_t i = 0; i < points.size(); ++i) {
            const auto& pt = points[i];

            // Calculate Range (Depth)
            float range = std::sqrt(pt.x*pt.x + pt.y*pt.y + pt.z*pt.z);
            if (range < 1e-3) continue;

            // Calculate angles
            float yaw = -std::atan2(pt.y, pt.x);
            float pitch = std::asin(pt.z / range);

            // Map to Pixel Coords (u, v)
            float proj_x = 0.5f * (yaw / M_PI + 1.0f); // [0, 1]
            float proj_y = 1.0f - (pitch + std::abs(fov_down)) / fov; // [0, 1]

            proj_x *= W;
            proj_y *= H;

            // Clamp
            int u = std::min(W - 1, std::max(0, (int)proj_x));
            int v = std::min(H - 1, std::max(0, (int)proj_y));

            int idx = v * W + u;

            // Store Point Index mapping
            // Note: If multiple points map to same pixel, we keep the closest one
            // or just overwrite (simple approach)
            proj_idx_map_[idx] = (int)i;

            // Fill Input Tensor (Planar 5xHxW)
            // Channel 0: Range
            data[0 * plane_size + idx] = range;
            // Channel 1: X
            data[1 * plane_size + idx] = pt.x;
            // Channel 2: Y
            data[2 * plane_size + idx] = pt.y;
            // Channel 3: Z
            data[3 * plane_size + idx] = pt.z;
            // Channel 4: Intensity
            data[4 * plane_size + idx] = pt.intensity;
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

PointCloudSegmenter::PointCloudSegmenter(const PointCloudSegConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PointCloudSegmenter::~PointCloudSegmenter() = default;
PointCloudSegmenter::PointCloudSegmenter(PointCloudSegmenter&&) noexcept = default;
PointCloudSegmenter& PointCloudSegmenter::operator=(PointCloudSegmenter&&) noexcept = default;

std::vector<SegmentedPoint> PointCloudSegmenter::segment(const std::vector<PointXYZI>& points) {
    if (!pimpl_) throw std::runtime_error("PointCloudS