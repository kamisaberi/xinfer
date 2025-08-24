#pragma once

#include <string>
#include <vector>
#include <memory>
#include <include/core/tensor.h>

namespace xinfer::zoo::threed {

    struct PointCloudSegmenterConfig {
        std::string engine_path;
        std::string labels_path = "";
    };

    class PointCloudSegmenter {
    public:
        explicit PointCloudSegmenter(const PointCloudSegmenterConfig& config);
        ~PointCloudSegmenter();

        PointCloudSegmenter(const PointCloudSegmenter&) = delete;
        PointCloudSegmenter& operator=(const PointCloudSegmenter&) = delete;
        PointCloudSegmenter(PointCloudSegmenter&&) noexcept;
        PointCloudSegmenter& operator=(PointCloudSegmenter&&) noexcept;

        std::vector<int> predict(const core::Tensor& point_cloud);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::threed

