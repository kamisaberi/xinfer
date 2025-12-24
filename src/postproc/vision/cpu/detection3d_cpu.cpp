#include "detection3d_cpu.h"
#include <xinfer/core/logging.h>
#include <cmath>
#include <algorithm>
#include <cstring>

namespace xinfer::postproc {

CpuDetection3DPostproc::CpuDetection3DPostproc() {}
CpuDetection3DPostproc::~CpuDetection3DPostproc() {}

void CpuDetection3DPostproc::init(const Detection3DConfig& config) {
    m_config = config;
    
    // Cache coordinate calculation constants
    m_min_x = m_config.pc_range[0];
    m_min_y = m_config.pc_range[1];
    
    // Each pixel in the output feature map represents (voxel_size * stride) meters
    m_out_size_factor_x = m_config.voxel_size_x * m_config.downsample_ratio;
    m_out_size_factor_y = m_config.voxel_size_y * m_config.downsample_ratio;
}

// Helper: Calculate Intersection over Union (IoU) in 2D BEV
// Simplified: Treats boxes as circles for rotation speed, or Axis-Aligned if rotation is small.
// For robust 3D NMS, one usually needs 'Polygonal Intersection' (Sutherland-Hodgman), 
// but that is very heavy for CPU.
// Here we use a Distance-based suppression for speed (common in Radar/Drone tracking).
static bool should_suppress(const BoundingBox3D& a, const BoundingBox3D& b, float thresh_dist) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dist_sq = dx*dx + dy*dy;
    
    // Heuristic: If centers are very close, suppress.
    // Threshold is derived from object size (e.g., 2m for a drone/car).
    float radius_sum = (std::max(a.w, a.l) + std::max(b.w, b.l)) / 2.0f;
    float dist_thresh = radius_sum * (1.0f - thresh_dist); // Approx conversion from IoU
    
    return dist_sq < (dist_thresh * dist_thresh);
}

void CpuDetection3DPostproc::apply_nms_bev(std::vector<BoundingBox3D>& boxes, float thresh) {
    if (boxes.empty()) return;

    // 1. Sort by score descending
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox3D& a, const BoundingBox3D& b) {
        return a.score > b.score;
    });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BoundingBox3D> keep;
    keep.reserve(std::min((size_t)m_config.max_detections, boxes.size()));

    // 2. Greedy NMS
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        
        keep.push_back(boxes[i]);
        if (keep.size() >= m_config.max_detections) break;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            
            // Optimization: Only compare same class
            if (boxes[i].class_id == boxes[j].class_id) {
                if (should_suppress(boxes[i], boxes[j], thresh)) {
                    suppressed[j] = true;
                }
            }
        }
    }
    boxes = std::move(keep);
}

std::vector<BoundingBox3D> CpuDetection3DPostproc::process(const std::vector<core::Tensor>& tensors) {
    std::vector<BoundingBox3D> proposals;

    // CenterPoint typically outputs:
    // Tensor 0: Heatmap [1, NumClasses, H, W]
    // Tensor 1: Regression [1, RegChannels, H, W] 
    //           (RegChannels often includes: dx, dy, z, log_w, log_l, log_h, sin, cos, vx, vy)
    
    if (tensors.size() < 2) {
        XINFER_LOG_ERROR("Detection3D expects at least 2 tensors (Heatmap, Regression).");
        return proposals;
    }

    const auto& hm = tensors[0];  // Heatmap
    const auto& reg = tensors[1]; // Regression

    auto hm_shape = hm.shape(); // [1, C, H, W]
    int num_classes = (int)hm_shape[1];
    int height = (int)hm_shape[2];
    int width = (int)hm_shape[3];
    int area = height * width;

    // Get Raw Pointers
    const float* hm_data = static_cast<const float*>(hm.data());
    const float* reg_data = static_cast<const float*>(reg.data());
    
    // Regression Layout Assumption: [1, Channels, H, W] (Planar)
    // Channel offsets in the regression map:
    // 0: offset_x, 1: offset_y, 2: z, 3: w, 4: l, 5: h, 6: sin, 7: cos, ...
    int reg_channels = (int)reg.shape()[1];
    int reg_plane_stride = area;

    // 1. Scan Heatmap for Peaks
    for (int c = 0; c < num_classes; ++c) {
        const float* class_map = hm_data + (c * area);
        
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                int idx = h * width + w;
                float score = class_map[idx];

                if (score > m_config.score_threshold) {
                    // Possible Detection
                    
                    // 2. Decode Regression Parameters
                    // Read across channels at spatial position 'idx'
                    float ox = reg_data[0 * reg_plane_stride + idx];
                    float oy = reg_data[1 * reg_plane_stride + idx];
                    float z  = reg_data[2 * reg_plane_stride + idx];
                    float dw = reg_data[3 * reg_plane_stride + idx];
                    float dl = reg_data[4 * reg_plane_stride + idx];
                    float dh = reg_data[5 * reg_plane_stride + idx];
                    
                    float rot_sin = reg_data[6 * reg_plane_stride + idx];
                    float rot_cos = reg_data[7 * reg_plane_stride + idx];

                    // 3. Coordinate Transformation (Grid -> Real World)
                    // x = (w + offset_x) * voxel_x + min_x
                    float cx = (w + ox) * m_out_size_factor_x + m_min_x;
                    float cy = (h + oy) * m_out_size_factor_y + m_min_y;
                    
                    // Dimensions usually exp(d)
                    float dim_w = std::exp(dw);
                    float dim_l = std::exp(dl);
                    float dim_h = std::exp(dh);

                    // Yaw calculation
                    float yaw = std::atan2(rot_sin, rot_cos);

                    BoundingBox3D box;
                    box.x = cx;
                    box.y = cy;
                    box.z = z;
                    box.w = dim_w;
                    box.l = dim_l;
                    box.h = dim_h;
                    box.yaw = yaw;
                    box.score = score;
                    box.class_id = c;
                    
                    // Velocity (Optional, if channels exist)
                    if (reg_channels >= 10) {
                        box.velocity_x = reg_data[8 * reg_plane_stride + idx];
                        box.velocity_y = reg_data[9 * reg_plane_stride + idx];
                    }

                    proposals.push_back(box);
                }
            }
        }
    }

    // 4. NMS
    apply_nms_bev(proposals, m_config.nms_threshold);

    return proposals;
}

} // namespace xinfer::postproc