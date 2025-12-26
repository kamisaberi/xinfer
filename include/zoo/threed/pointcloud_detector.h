#pragma once

#include <string>
#include <vector>
#include <memory>
#include <array>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/detection3d_interface.h> // For BoundingBox3D

namespace xinfer::zoo::threed {

    /**
     * @brief A single 3D point (LiDAR format).
     */
    struct PointXYZI {
        float x, y, z;
        float intensity;
    };

    /**
     * @brief Configuration for 3D Detector.
     */
    struct PointCloudConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT; // 3D models are heavy, GPU preferred

        // Model Path (e.g., pointpillars.engine)
        std::string model_path;

        // Voxelization Settings (Must match model training)
        std::array<float, 6> pc_range = {-51.2f, -51.2f, -5.0f, 51.2f, 51.2f, 3.0f};
        float voxel_size_x = 0.16f;
        float voxel_size_y = 0.16f;
        float voxel_size_z = 4.0f; // Pillars usually have z=range

        int max_points_per_voxel = 32;
        int max_voxels = 40000;

        // Post-processing thresholds
        float score_threshold = 0.3f;
        float nms_threshold = 0.1f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PointCloudDetector {
    public:
        explicit PointCloudDetector(const PointCloudConfig& config);
        ~PointCloudDetector();

        // Move semantics
        PointCloudDetector(PointCloudDetector&&) noexcept;
        PointCloudDetector& operator=(PointCloudDetector&&) noexcept;
        PointCloudDetector(const PointCloudDetector&) = delete;
        PointCloudDetector& operator=(const PointCloudDetector&) = delete;

        /**
         * @brief Detect objects in a point cloud.
         *
         * Pipeline:
         * 1. Voxelization (CPU or CUDA) -> Converts points to Pillars.
         * 2. Feature Extraction (Pillar Feature Net).
         * 3. Backbone (2D CNN).
         * 4. Detection Head & Post-processing (3D Decoding + NMS).
         *
         * @param points Raw point cloud data.
         * @return List of 3D Bounding Boxes.
         */
        std::vector<postproc::BoundingBox3D> detect(const std::vector<PointXYZI>& points);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed