#include <xinfer/zoo/aec/site_safety_monitor.h>
#include <xinfer/core/logging.h>

// --- Reuse other Zoo modules ---
#include <xinfer/zoo/vision/detector.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <map>

namespace xinfer::zoo::aec {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SiteSafetyMonitor::Impl {
    SafetyConfig config_;

    // --- Components ---
    std::unique_ptr<vision::ObjectDetector> ppe_detector_;
    std::unique_ptr<vision::ObjectDetector> machinery_detector_;
    std::unique_ptr<postproc::ITracker> person_tracker_;

    // --- Class ID Mappings ---
    int id_person = -1, id_hardhat = -1, id_vest = -1;

    Impl(const SafetyConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Init PPE Detector
        vision::DetectorConfig ppe_cfg;
        ppe_cfg.target = config_.target;
        ppe_cfg.model_path = config_.ppe_model_path;
        ppe_cfg.labels_path = config_.ppe_labels_path;
        ppe_cfg.confidence_threshold = config_.ppe_conf_thresh;
        ppe_detector_ = std::make_unique<vision::ObjectDetector>(ppe_cfg);

        // Map labels to IDs
        // (Assuming a getter or public member for labels in ObjectDetector for this)
        // For simplicity, we hardcode: 0=Person, 1=Hardhat, 2=Vest
        id_person = 0; id_hardhat = 1; id_vest = 2;

        // 2. Init Machinery Detector
        vision::DetectorConfig mach_cfg;
        mach_cfg.target = config_.target;
        mach_cfg.model_path = config_.machinery_model_path;
        mach_cfg.labels_path = config_.machinery_labels_path;
        mach_cfg.confidence_threshold = config_.machinery_conf_thresh;
        machinery_detector_ = std::make_unique<vision::ObjectDetector>(mach_cfg);

        // 3. Init Person Tracker
        person_tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        person_tracker_->init(trk_cfg);
    }

    // Helper: IoU to check if one box is inside another
    float box_iou(const cv::Rect& a, const cv::Rect& b) {
        float inter_area = (float)(a & b).area();
        float union_area = (float)a.area() + (float)b.area() - inter_area;
        return inter_area / (union_area + 1e-6f);
    }
};

// =================================================================================
// Public API
// =================================================================================

SiteSafetyMonitor::SiteSafetyMonitor(const SafetyConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SiteSafetyMonitor::~SiteSafetyMonitor() = default;
SiteSafetyMonitor::SiteSafetyMonitor(SiteSafetyMonitor&&) noexcept = default;
SiteSafetyMonitor& SiteSafetyMonitor::operator=(SiteSafetyMonitor&&) noexcept = default;

SiteStatus SiteSafetyMonitor::monitor(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SiteSafetyMonitor is null.");

    SiteStatus result;
    result.annotated_image = image.clone();
    result.is_safe = true;

    // 1. Run Detectors
    auto ppe_dets = pimpl_->ppe_detector_->predict(image);
    auto mach_dets = pimpl_->machinery_detector_->predict(image);

    // 2. Separate detections by class
    std::vector<vision::BoundingBox> persons, hardhats, vests;
    for (const auto& d : ppe_dets) {
        if (d.class_id == pimpl_->id_person) persons.push_back(d);
        else if (d.class_id == pimpl_->id_hardhat) hardhats.push_back(d);
        else if (d.class_id == pimpl_->id_vest) vests.push_back(d);
    }

    // 3. Track Persons
    auto tracks = pimpl_->person_tracker_->update(persons);

    // 4. Rule Engine
    for (const auto& p_track : tracks) {
        bool has_hat = false;
        bool has_vest = false;

        cv::Rect person_box(p_track.box.x1, p_track.box.y1, p_track.box.x2-p_track.box.x1, p_track.box.y2-p_track.box.y1);

        // A. Check for PPE
        for (const auto& h : hardhats) {
            cv::Rect hat_box(h.x1, h.y1, h.x2-h.x1, h.y2-h.y1);
            if ((person_box & hat_box).area() > 0) { // Simple overlap check
                has_hat = true;
                break;
            }
        }

        // Similar check for vests
        // ...

        if (!has_hat) {
            result.is_safe = false;
            result.alerts.push_back({p_track.track_id, SafetyViolation::MISSING_HARD_HAT, p_track.box, p_track.box.confidence});
        }

        // B. Check for Proximity to Machinery
        for (const auto& m : mach_dets) {
            cv::Point person_center(person_box.x + person_box.width/2, person_box.y + person_box.height/2);
            cv::Rect mach_box(m.x1, m.y1, m.x2-m.x1, m.y2-m.y1);
            cv::Point mach_center(mach_box.x + mach_box.width/2, mach_box.y + mach_box.height/2);

            float dist_px = (float)cv::norm(person_center - mach_center);
            float dist_m = dist_px / pimpl_->config_.pixels_per_meter;

            if (dist_m < pimpl_->config_.safe_distance_m) {
                result.is_safe = false;
                result.alerts.push_back({p_track.track_id, SafetyViolation::PROXIMITY_BREACH, p_track.box, p_track.box.confidence});
            }
        }
    }

    // 5. Visualization
    for (const auto& alert : result.alerts) {
        cv::Rect r(alert.box.x1, alert.box.y1, alert.box.x2-alert.box.x1, alert.box.y2-alert.box.y1);
        cv::rectangle(result.annotated_image, r, cv::Scalar(0,0,255), 3);

        std::string text;
        if (alert.violation == SafetyViolation::MISSING_HARD_HAT) text = "NO HELMET";
        if (alert.violation == SafetyViolation::PROXIMITY_BREACH) text = "TOO CLOSE";

        cv::putText(result.annotated_image, text, r.tl(), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,255), 3);
    }

    return result;
}

} // namespace xinfer::zoo::aec```