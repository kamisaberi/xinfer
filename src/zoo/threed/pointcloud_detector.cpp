#include <include/zoo/threed/pointcloud_detector.h>
#include <stdexcept>
#include <fstream>
#include <vector>

#include <include/core/engine.h>

namespace xinfer::zoo::threed {

struct PointCloudDetector::Impl {
    PointCloudDetectorConfig config_;
    std::unique_ptr<core::InferenceEngine> engine_;
    std::vector<std::string> class_labels_;
};

PointCloudDetector::PointCloudDetector(const PointCloudDetectorConfig& config)
    : pimpl_(new Impl{config})
{
    if (!std::ifstream(pimpl_->config_.engine_path).good()) {
        throw std::runtime_error("PointCloudDetector engine file not found: " + pimpl_->config_.engine_path);
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

PointCloudDetector::~PointCloudDetector() = default;
PointCloudDetector::PointCloudDetector(PointCloudDetector&&) noexcept = default;
PointCloudDetector& PointCloudDetector::operator=(PointCloudDetector&&) noexcept = default;

std::vector<BoundingBox3D> PointCloudDetector::predict(const core::Tensor& point_cloud) {
    if (!pimpl_) throw std::runtime_error("PointCloudDetector is in a moved-from state.");

    auto output_tensors = pimpl_->engine_->infer({point_cloud});

    const core::Tensor& boxes_tensor = output_tensors[0];
    const core::Tensor& scores_tensor = output_tensors[1];
    const core::Tensor& labels_tensor = output_tensors[2];

    std::vector<float> h_boxes(boxes_tensor.num_elements());
    std::vector<float> h_scores(scores_tensor.num_elements());
    std::vector<int> h_labels(labels_tensor.num_elements());

    boxes_tensor.copy_to_host(h_boxes.data());
    scores_tensor.copy_to_host(h_scores.data());
    labels_tensor.copy_to_host(h_labels.data());

    std::vector<BoundingBox3D> results;
    int num_detections = scores_tensor.shape()[0];

    for (int i = 0; i < num_detections; ++i) {
        if (h_scores[i] < pimpl_->config_.score_threshold) {
            continue;
        }

        BoundingBox3D box;
        box.confidence = h_scores[i];
        box.class_id = h_labels[i];

        box.x = h_boxes[i * 7 + 0];
        box.y = h_boxes[i * 7 + 1];
        box.z = h_boxes[i * 7 + 2];
        box.length = h_boxes[i * 7 + 3];
        box.width = h_boxes[i * 7 + 4];
        box.height = h_boxes[i * 7 + 5];
        box.yaw = h_boxes[i * 7 + 6];

        if (!pimpl_->class_labels_.empty() && box.class_id < pimpl_->class_labels_.size()) {
            box.label = pimpl_->class_labels_[box.class_id];
        } else {
            box.label = "Class " + std::to_string(box.class_id);
        }

        results.push_back(box);
    }

    return results;
}

} // namespace xinfer::zoo::threed