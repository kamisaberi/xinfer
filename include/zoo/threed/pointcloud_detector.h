#pragma once

#include <string>
#include <vector>
#include <memory>

namespace xinfer::core { class Tensor; }

namespace xinfer::zoo::threed {

    struct BoundingBox3D {
        int class_id;
        float confidence;
        std::string label;
        float x, y, z; // Center coordinates
        float length, width, height;
        float yaw; // Rotation around Z-axis
    };

    struct PointCloudDetectorConfig {
        std::string engine_path;
        std::string labels_path = "";
        float score_threshold = 0.5f;
    };

    class PointCloudDetector {
    public:
        explicit PointCloudDetector(const PointCloudDetectorConfig& config);
        ~PointCloudDetector();

        PointCloudDetector(const PointCloudDetector&) = delete;
        PointCloudDetector& operator=(const PointCloudDetector&) = delete;
        PointCloudDetector(PointCloudDetector&&) noexcept;
        PointCloudDetector& operator=(PointCloudDetector&&) noexcept;

        std::vector<BoundingBox3D> predict(const core::Tensor& point_cloud);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed

