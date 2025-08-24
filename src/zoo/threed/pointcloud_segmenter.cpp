#include <include/zoo/threed/pointcloud_segmenter.h>
#include <stdexcept>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>

#include <include/core/engine.h>

namespace xinfer::zoo::threed {

struct PointCloudSegmenter::Impl {
    PointCloudSegmenterConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::vector<std::string> class_labels_;
};

PointCloudSegmenter::PointCloudSegmenter(const PointCloudSegmenterConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("PointCloudSegmenter engine file not found: " + pimpl_->config_.engine_path);
    }

    pimpl_->engine_ = std::make_unique<core::InferenceEngine>(pimpl_->config_.engine_path);

    if (!pimpl_->config_.labels_path.empty()) {
        std::ifstream labels_file(pimpl_->config_.labels_path);
        if (!labels_file) throw std::runtime_error("Could not open labels file: " + pimpl_->config_.labels_path);
        std::string line;
        while (std::getline(labels_file, line)) {
            pimpl_->class_labels_.push_back(line);
        }
    }
}

PointCloudSegmenter::~PointCloudSegmenter() = default;
PointCloudSegmenter::PointCloudSegmenter(PointCloudSegmenter&&) noexcept = default;
PointCloudSegmenter& PointCloudSegmenter::operator=(PointCloudSegmenter&&) noexcept = default;

std::vector<int> PointCloudSegmenter::predict(const core::Tensor& point_cloud) {
    if (!pimpl_) throw std::runtime_error("PointCloudSegmenter is in a moved-from state.");

    auto output_tensors = pimpl_->engine_->infer({point_cloud});

    const core::Tensor& logits_tensor = output_tensors[0];

    auto logits_shape = logits_tensor.shape();
    const int num_points = logits_shape[0];
    const int num_classes = logits_shape[1];

    std::vector<float> h_logits(logits_tensor.num_elements());
    logits_tensor.copy_to_host(h_logits.data());

    std::vector<int> point_labels(num_points);

    for (int i = 0; i < num_points; ++i) {
        const float* point_logits = h_logits.data() + i * num_classes;
        auto max_it = std::max_element(point_logits, point_logits + num_classes);
        point_labels[i] = std::distance(point_logits, max_it);
    }

    return point_labels;
}

} // namespace xinfer::zoo::threed