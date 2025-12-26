#include <xinfer/zoo/threed/pointcloud_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Preproc factory skipped; Voxelization is specific to 3D and implemented internally here.
#include <xinfer/postproc/factory.h>

#include <iostream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <map>

namespace xinfer::zoo::threed {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PointCloudDetector::Impl {
    PointCloudConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<postproc::IDetection3DPostprocessor> postproc_;

    // Data Containers
    // PointPillars inputs usually:
    // 1. features: [1, MaxVoxels, MaxPoints, NumFeats] (e.g. x,y,z,i,xc,yc,zc,xp,yp)
    // 2. indices:  [1, MaxVoxels, 4] (Batch, z, y, x coords)
    // 3. num_points: [1, MaxVoxels] (Actual points per pillar)
    core::Tensor t_voxels;
    core::Tensor t_coords;
    core::Tensor t_num_points;

    // Outputs
    core::Tensor t_cls_score; // Heatmap
    core::Tensor t_bbox_pred; // Regression
    core::Tensor t_dir_cls;   // Direction (optional)

    // Derived Constants
    int grid_w, grid_h, grid_d;

    Impl(const PointCloudConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Calculate Grid Dimensions
        grid_w = std::floor((config_.pc_range[3] - config_.pc_range[0]) / config_.voxel_size_x);
        grid_h = std::floor((config_.pc_range[4] - config_.pc_range[1]) / config_.voxel_size_y);
        grid_d = std::floor((config_.pc_range[5] - config_.pc_range[2]) / config_.voxel_size_z);

        // 2. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PointCloudDetector: Failed to load model " + config_.model_path);
        }

        // 3. Setup 3D Post-processor
        postproc_ = postproc::create_detection3d(config_.target);

        postproc::Detection3DConfig det_cfg;
        det_cfg.score_threshold = config_.score_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        det_cfg.voxel_size_x = config_.voxel_size_x;
        det_cfg.voxel_size_y = config_.voxel_size_y;
        det_cfg.pc_range = config_.pc_range;
        // PointPillars backbone usually downsamples by 2 or 4.
        // We assume 2 here, or it should be configurable.
        det_cfg.downsample_ratio = 2;

        postproc_->init(det_cfg);

        // 4. Pre-allocate Input Tensors
        // Assume 9 input features per point: x, y, z, i, x_c, y_c, z_c, x_p, y_p
        int num_point_feats = 9;

        t_voxels.resize({1, (int64_t)config_.max_voxels, (int64_t)config_.max_points_per_voxel, num_point_feats}, core::DataType::kFLOAT);
        t_coords.resize({1, (int64_t)config_.max_voxels, 4}, core::DataType::kINT32); // [b, z, y, x]
        t_num_points.resize({1, (int64_t)config_.max_voxels}, core::DataType::kFLOAT); // Some models use int, some float
    }

    // --- CPU Voxelization (Simplified PointPillars Preprocessing) ---
    void voxelize_cpu(const std::vector<PointXYZI>& points) {
        // Clear tensors
        float* vox_data = static_cast<float*>(t_voxels.data());
        int32_t* coor_data = static_cast<int32_t*>(t_coords.data());
        float* num_data = static_cast<float*>(t_num_points.data());

        // Zero out
        std::memset(vox_data, 0, t_voxels.size() * sizeof(float));
        std::memset(coor_data, -1, t_coords.size() * sizeof(int32_t)); // -1 init
        std::memset(num_data, 0, t_num_points.size() * sizeof(float));

        // Maps grid index (y*W + x) -> Voxel Index (0..MaxVoxels)
        std::map<int, int> grid_to_voxel;
        int voxel_count = 0;

        float range_min_x = config_.pc_range[0];
        float range_min_y = config_.pc_range[1];
        float range_min_z = config_.pc_range[2];

        float inv_vx = 1.0f / config_.voxel_size_x;
        float inv_vy = 1.0f / config_.voxel_size_y;
        float inv_vz = 1.0f / config_.voxel_size_z;

        for (const auto& pt : points) {
            if (pt.x < config_.pc_range[0] || pt.x >= config_.pc_range[3] ||
                pt.y < config_.pc_range[1] || pt.y >= config_.pc_range[4] ||
                pt.z < config_.pc_range[2] || pt.z >= config_.pc_range[5]) {
                continue; // Out of range
            }

            int coor_x = (int)((pt.x - range_min_x) * inv_vx);
            int coor_y = (int)((pt.y - range_min_y) * inv_vy);
            int coor_z = (int)((pt.z - range_min_z) * inv_vz);

            int grid_idx = coor_y * grid_w + coor_x;
            int v_idx = -1;

            auto it = grid_to_voxel.find(grid_idx);
            if (it == grid_to_voxel.end()) {
                if (voxel_count >= config_.max_voxels) continue; // Full
                v_idx = voxel_count++;
                grid_to_voxel[grid_idx] = v_idx;

                // Set coordinates [b, z, y, x]
                coor_data[v_idx * 4 + 0] = 0; // Batch 0
                coor_data[v_idx * 4 + 1] = coor_z;
                coor_data[v_idx * 4 + 2] = coor_y;
                coor_data[v_idx * 4 + 3] = coor_x;
            } else {
                v_idx = it->second;
            }

            // Add point to voxel
            int num = (int)num_data[v_idx];
            if (num < config_.max_points_per_voxel) {
                // Feature offset: v_idx * (MaxPts * 9) + num * 9
                float* feat_ptr = vox_data + (v_idx * config_.max_points_per_voxel * 9) + (num * 9);

                feat_ptr[0] = pt.x;
                feat_ptr[1] = pt.y;
                feat_ptr[2] = pt.z;
                feat_ptr[3] = pt.intensity;
                // Arithmetic features (x_c, y_c, z_c, x_p, y_p) usually calculated here
                // Calculated as (pt - mean_of_voxel) and (pt - center_of_voxel)
                // Simplified for brevity:
                feat_ptr[4] = pt.x - (coor_x * config_.voxel_size_x + range_min_x + config_.voxel_size_x/2);
                feat_ptr[5] = pt.y - (coor_y * config_.voxel_size_y + range_min_y + config_.voxel_size_y/2);
                feat_ptr[6] = pt.z - (coor_z * config_.voxel_size_z + range_min_z + config_.voxel_size_z/2);
                feat_ptr[7] = 0.0f; // Placeholder x_p (x - voxel_center)
                feat_ptr[8] = 0.0f; // Placeholder y_p (y - voxel_center)

                num_data[v_idx] += 1.0f;
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

PointCloudDetector::PointCloudDetector(const PointCloudConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PointCloudDetector::~PointCloudDetector() = default;
PointCloudDetector::PointCloudDetector(PointCloudDetector&&) noexcept = default;
PointCloudDetector& PointCloudDetector::operator=(PointCloudDetector&&) noexcept = default;

std::vector<postproc::BoundingBox3D> PointCloudDetector::detect(const std::vector<PointXYZI>& points) {
    if (!pimpl_) throw std::runtime_error("PointCloudDetector is null.");

    // 1. Voxelization (CPU)
    // Note: For extreme performance on Jetson, you should move this to a CUDA kernel.
    pimpl_->voxelize_cpu(points);

    // 2. Inference
    // PointPillars Inputs: [Voxels, NumPoints, Coords]
    pimpl_->engine_->predict(
        {pimpl_->t_voxels, pimpl_->t_num_points, pimpl_->t_coords},
        {pimpl_->t_cls_score, pimpl_->t_bbox_pred, pimpl_->t_dir_cls}
    );

    // 3. Post-processing (3D Decode & NMS)
    // Supports CenterPoint/PointPillars outputs
    return pimpl_->postproc_->process({pimpl_->t_cls_score, pimpl_->t_bbox_pred});
}

} // namespace xinfer::zoo::threed