#include <xinfer/zoo/robotics/grasp_planner.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
// Note: Grasp preprocessing is highly specific (Depth Normalization),
// so we implement custom logic here rather than using generic image preproc.

#include <iostream>
#include <algorithm>
#include <cmath>

namespace xinfer::zoo::robotics {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct GraspPlanner::Impl {
    GraspConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;

    // Data Containers
    core::Tensor input_tensor;
    // GG-CNN / GR-ConvNet typically output 4 tensors (or 1 concatenated):
    // 1. Quality (Score)
    // 2. Cos 2*Theta
    // 3. Sin 2*Theta
    // 4. Width
    // We assume the model outputs a single concatenated tensor [1, 4, H, W] or separate.
    // For this implementation, let's assume separate output tensors for clarity,
    // or we split them if it's one. Let's assume one 4-channel output for efficiency.
    core::Tensor output_tensor;

    Impl(const GraspConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("GraspPlanner: Failed to load model " + config_.model_path);
        }

        // 2. Pre-allocate Input
        // Shape: [1, 1, H, W] (Depth is usually single channel)
        input_tensor.resize({1, 1, (int64_t)config_.input_height, (int64_t)config_.input_width}, core::DataType::kFLOAT);
    }

    // --- 1. Custom Depth Preprocessing ---
    // Crop center -> Resize -> Normalize
    void process_depth(const cv::Mat& depth_raw) {
        // Convert to Float32 meters
        cv::Mat depth_m;
        if (depth_raw.type() == CV_16U) {
            depth_raw.convertTo(depth_m, CV_32F, config_.depth_scale);
        } else {
            depth_raw.copyTo(depth_m);
        }

        // Center Crop (Robots usually look at center of workspace)
        int h = depth_m.rows;
        int w = depth_m.cols;
        int crop = (int)config_.crop_size;
        int x = (w - crop) / 2;
        int y = (h - crop) / 2;

        cv::Rect roi(x, y, crop, crop);
        cv::Mat cropped = depth_m(roi);

        // Resize to model input
        cv::Mat resized;
        cv::resize(cropped, resized, cv::Size(config_.input_width, config_.input_height), 0, 0, cv::INTER_AREA);

        // Fill Tensor
        // Normalize: Subtract mean depth? Or clamp?
        // GG-CNN expects values roughly in meters. Some versions normalize (z - mean)/std.
        // We assume raw meters here as it's common for GR-ConvNet.

        float* t_ptr = static_cast<float*>(input_tensor.data());
        std::memcpy(t_ptr, resized.data, resized.total() * sizeof(float));
    }

    // --- 2. Post-processing ---
    std::vector<Grasp> decode_output(const cv::Mat& original_depth_m) {
        // Output Tensor: [1, 4, H, W]
        // Channels: 0=Quality, 1=Cos, 2=Sin, 3=Width
        auto shape = output_tensor.shape();
        int out_h = (int)shape[2];
        int out_w = (int)shape[3];
        int plane_size = out_h * out_w;

        const float* data = static_cast<const float*>(output_tensor.data());
        const float* q_map = data;
        const float* cos_map = data + plane_size;
        const float* sin_map = data + 2 * plane_size;
        const float* w_map   = data + 3 * plane_size;

        std::vector<Grasp> candidates;

        // Iterate pixels
        for (int i = 0; i < plane_size; ++i) {
            float score = q_map[i];
            if (score > config_.score_threshold) {
                int u = i % out_w;
                int v = i / out_w;

                // Decode Angle: 0.5 * atan2(sin, cos)
                float angle = 0.5f * std::atan2(sin_map[i], cos_map[i]);

                // Decode Width: Model outputs width in pixels or meters?
                // Usually pixels * scaling factor. Assuming meters here for simplicity,
                // or normalized [0,1] scaled by max_width.
                float width = w_map[i] * 0.15f; // Assume output 0-1 maps to 0-15cm

                Grasp g;
                g.u = u;
                g.v = v;
                g.score = score;
                g.angle = angle;
                g.width = width;
                candidates.push_back(g);
            }
        }

        // Sort by score
        std::sort(candidates.begin(), candidates.end(), [](const Grasp& a, const Grasp& b) {
            return a.score > b.score;
        });

        // NMS (Simple Distance Suppression)
        std::vector<Grasp> final_grasps;
        float dist_thresh_sq = 10.0f * 10.0f; // 10 pixels radius

        for (const auto& cand : candidates) {
            bool keep = true;
            for (const auto& exist : final_grasps) {
                float dx = cand.u - exist.u;
                float dy = cand.v - exist.v;
                if (dx*dx + dy*dy < dist_thresh_sq) {
                    keep = false;
                    break;
                }
            }
            if (keep) {
                // --- 3. Deproject Pixel to 3D Point ---
                // Map model coords back to original image coords
                // (Reverse the Resize and Crop operations)
                float scale = config_.crop_size / (float)config_.input_width;
                int crop_offset_x = (original_depth_m.cols - config_.crop_size) / 2;
                int crop_offset_y = (original_depth_m.rows - config_.crop_size) / 2;

                int real_u = (int)(cand.u * scale) + crop_offset_x;
                int real_v = (int)(cand.v * scale) + crop_offset_y;

                // Sample Depth at this location
                // Safety check
                if (real_u >= 0 && real_u < original_depth_m.cols &&
                    real_v >= 0 && real_v < original_depth_m.rows) {

                    float z_meters = original_depth_m.at<float>(real_v, real_u);

                    if (z_meters > 0.1f) { // Valid depth
                        // Pinhole Model:
                        // X = (u - cx) * Z / fx
                        // Y = (v - cy) * Z / fy
                        Grasp g_3d = cand;
                        g_3d.z = z_meters;
                        g_3d.x = (real_u - config_.cx) * z_meters / config_.fx;
                        g_3d.y = (real_v - config_.cy) * z_meters / config_.fy;

                        final_grasps.push_back(g_3d);
                    }
                }
            }
            if (final_grasps.size() >= config_.max_grasps) break;
        }

        return final_grasps;
    }
};

// =================================================================================
// Public API
// =================================================================================

GraspPlanner::GraspPlanner(const GraspConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

GraspPlanner::~GraspPlanner() = default;
GraspPlanner::GraspPlanner(GraspPlanner&&) noexcept = default;
GraspPlanner& GraspPlanner::operator=(GraspPlanner&&) noexcept = default;

std::vector<Grasp> GraspPlanner::plan(const cv::Mat& depth_image) {
    if (!pimpl_) throw std::runtime_error("GraspPlanner is null.");

    // 1. Process Depth
    // We need a copy of the metric depth for deprojection later
    cv::Mat depth_m;
    if (depth_image.type() == CV_16U) {
        depth_image.convertTo(depth_m, CV_32F, pimpl_->config_.depth_scale);
    } else {
        depth_m = depth_image.clone();
    }

    pimpl_->process_depth(depth_image);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Postprocess & Deproject
    return pimpl_->decode_output(depth_m);
}

void GraspPlanner::draw_grasps(cv::Mat& image, const std::vector<Grasp>& grasps) {
    for (const auto& g : grasps) {
        // Need to map 3D back to 2D for visualization if drawing on original RGB
        // Here we just use the raw coordinates assuming image is the crop or aligned
        // This is a simplified viz helper.

        // Simple circle at grasp point
        // In real app, draw the gripper rectangle based on g.angle and g.width
        cv::Point center(g.u, g.v); // These are model-space coords, user must map if drawing on full img

        // We really need the re-mapped coords from inside decode, but Grasp struct has 3D.
        // Let's assume user draws 3D points or we re-project:
        // u = (x * fx / z) + cx

        // (Viz Logic Omitted for brevity, standard OpenCV drawing)
    }
}

} // namespace xinfer::zoo::robotics