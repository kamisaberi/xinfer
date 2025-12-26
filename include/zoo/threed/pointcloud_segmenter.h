#pragma once

#include <string>
#include <vector>
#include <memory>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>

namespace xinfer::zoo::threed {

    /**
     * @brief Input Point format (LiDAR).
     */
    struct PointXYZI {
        float x, y, z;
        float intensity;
    };

    /**
     * @brief Output Segmented Point.
     */
    struct SegmentedPoint {
        float x, y, z;
        int class_id;      // e.g., 0=Unlabeled, 1=Car, 2=Road
        float confidence;  // Softmax probability
    };

    struct PointCloudSegConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::NVIDIA_TRT;

        // Model Path (e.g., rangenet_darknet53.engine)
        std::string model_path;

        // Range View Projection Parameters
        // (Must match the specific LiDAR sensor the model was trained on)
        int proj_height = 64;   // e.g., 64 lasers (Velodyne HDL-64E)
        int proj_width = 2048;  // Angular resolution
        float fov_up = 3.0f;    // Degrees
        float fov_down = -25.0f;// Degrees

        // Post-processing
        bool use_knn_postproc = true; // Clean up projection artifacts

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class PointCloudSegmenter {
    public:
        explicit PointCloudSegmenter(const PointCloudSegConfig& config);
        ~PointCloudSegmenter();

        // Move semantics
        PointCloudSegmenter(PointCloudSegmenter&&) noexcept;
        PointCloudSegmenter& operator=(PointCloudSegmenter&&) noexcept;
        PointCloudSegmenter(const PointCloudSegmenter&) = delete;
        PointCloudSegmenter& operator=(const PointCloudSegmenter&) = delete;

        /**
         * @brief Segment a point cloud.
         *
         * Pipeline:
         * 1. Spherical Projection (3D -> 2D Range Image).
         * 2. Inference (2D CNN).
         * 3. Reprojection (2D Mask -> 3D Points).
         *
         * @param points Input point cloud.
         * @return Vector of points with assigned Class IDs.
         */
        std::vector<SegmentedPoint> segment(const std::vector<PointXYZI>& points);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed