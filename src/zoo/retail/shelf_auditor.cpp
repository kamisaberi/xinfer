#include <include/zoo/retail/shelf_auditor.h>
#include <stdexcept>
#include <map>

#include <include/zoo/vision/detector.h>

namespace xinfer::zoo::retail {

struct ShelfAuditor::Impl {
    ShelfAuditorConfig config_;
    std::unique_ptr<vision::ObjectDetector> detector_;
};

ShelfAuditor::ShelfAuditor(const ShelfAuditorConfig& config)
    : pimpl_(new Impl{config})
{
    vision::DetectorConfig det_config;
    det_config.engine_path = pimpl_->config_.detection_engine_path;
    det_config.labels_path = pimpl_->config_.labels_path;
    det_config.confidence_threshold = pimpl_->config_.confidence_threshold;
    det_config.nms_iou_threshold = pimpl_->config_.nms_iou_threshold;
    det_config.input_width = pimpl_->config_.detection_input_width;
    det_config.input_height = pimpl_->config_.detection_input_height;

    pimpl_->detector_ = std::make_unique<vision::ObjectDetector>(det_config);
}

ShelfAuditor::~ShelfAuditor() = default;
ShelfAuditor::ShelfAuditor(ShelfAuditor&&) noexcept = default;
ShelfAuditor& ShelfAuditor::operator=(ShelfAuditor&&) noexcept = default;

std::vector<ShelfItem> ShelfAuditor::audit(const cv::Mat& shelf_image) {
    if (!pimpl_) throw std::runtime_error("ShelfAuditor is in a moved-from state.");

    auto detections = pimpl_->detector_->predict(shelf_image);

    std::map<int, ShelfItem> item_map;
    for (const auto& det : detections) {
        if (item_map.find(det.class_id) == item_map.end()) {
            ShelfItem new_item;
            new_item.class_id = det.class_id;
            new_item.label = det.label;
            new_item.count = 0;
            item_map[det.class_id] = new_item;
        }
        item_map[det.class_id].count++;
        cv::Rect box(cv::Point(det.x1, det.y1), cv::Point(det.x2, det.y2));
        item_map[det.class_id].locations.push_back(box);
    }

    std::vector<ShelfItem> results;
    for (auto const& [key, val] : item_map) {
        results.push_back(val);
    }

    return results;
}

} // namespace xinfer::zoo::retail