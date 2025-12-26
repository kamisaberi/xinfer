#include <xinfer/zoo/maritime/port_analyzer.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <fstream>
#include <map>
#include <set>

namespace xinfer::zoo::maritime {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct PortAnalyzer::Impl {
    PortConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // State
    // Map Track ID -> {Last Position, Dwell Frames}
    struct TrackState {
        cv::Rect last_box;
        int dwell_frames = 0;
    };
    std::map<int, TrackState> track_history_;

    // Labels
    std::vector<std::string> labels_;
    // Pre-calculated sets of IDs for fast lookup
    std::set<int> vessel_ids_;
    std::set<int> truck_ids_;
    std::set<int> container_ids_;

    Impl(const PortConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("PortAnalyzer: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Detector Post-processing
        detector_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        detector_->init(det_cfg);

        // 4. Setup Tracker
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = 60;
        trk_cfg.min_hits = 3;
        tracker_->init(trk_cfg);

        // 5. Load Labels and Classify
        load_labels(config_.labels_path);
        for (size_t i = 0; i < labels_.size(); ++i) {
            std::string l = labels_[i];
            // Simple string matching
            if (l.find("ship") != std::string::npos || l.find("boat") != std::string::npos) vessel_ids_.insert(i);
            if (l.find("truck") != std::string::npos) truck_ids_.insert(i);
            if (l.find("container") != std::string::npos) container_ids_.insert(i);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }

    // Helper: Check if a box center is inside a zone
    bool is_in_zone(const cv::Rect& box, const cv::Rect& zone) {
        cv::Point center = (box.tl() + box.br()) * 0.5;
        return zone.contains(center);
    }
};

// =================================================================================
// Public API
// =================================================================================

PortAnalyzer::PortAnalyzer(const PortConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

PortAnalyzer::~PortAnalyzer() = default;
PortAnalyzer::PortAnalyzer(PortAnalyzer&&) noexcept = default;
PortAnalyzer& PortAnalyzer::operator=(PortAnalyzer&&) noexcept = default;

PortAnalytics PortAnalyzer::analyze(const cv::Mat& image, float fps) {
    if (!pimpl_) throw std::runtime_error("PortAnalyzer is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect
    auto raw_dets = pimpl_->detector_->process({pimpl_->output_tensor});

    // Scale detections
    std::vector<postproc::BoundingBox> scaled_dets;
    float sx = (float)image.cols / pimpl_->config_.input_width;
    float sy = (float)image.rows / pimpl_->config_.input_height;

    for (auto& d : raw_dets) {
        d.x1 *= sx; d.x2 *= sx; d.y1 *= sy; d.y2 *= sy;
        scaled_dets.push_back(d);
    }

    // 4. Track
    auto tracks = pimpl_->tracker_->update(scaled_dets);

    // 5. Analytics
    PortAnalytics analytics = {0};
    analytics.overview_image = image.clone();

    // Convert normalized zones to pixel coordinates
    cv::Rect berth_zone(
        (int)(pimpl_->config_.berth_zone.x * image.cols),
        (int)(pimpl_->config_.berth_zone.y * image.rows),
        (int)(pimpl_->config_.berth_zone.width * image.cols),
        (int)(pimpl_->config_.berth_zone.height * image.rows)
    );
    // ... (repeat for other zones)

    float total_dwell_time = 0;
    int waiting_truck_count = 0;

    for (const auto& t : tracks) {
        cv::Rect box((int)t.box.x1, (int)t.box.y1, (int)(t.box.x2 - t.box.x1), (int)(t.box.y2 - t.box.y1));

        // Update Dwell Time
        auto& hist = pimpl_->track_history_[t.track_id];
        // Check if moved (IoU or center dist)
        if (hist.last_box.area() > 0) {
            float iou = (float)(hist.last_box & box).area() / (float)(hist.last_box | box).area();
            if (iou > 0.95f) {
                hist.dwell_frames++;
            } else {
                hist.dwell_frames = 0;
            }
        }
        hist.last_box = box;

        // Logic
        if (pimpl_->vessel_ids_.count(t.box.class_id) && pimpl_->is_in_zone(box, berth_zone)) {
            analytics.num_vessels_at_berth++;
        }
        if (pimpl_->truck_ids_.count(t.box.class_id) /* && is_in_zone(truck_queue_zone)*/) {
            if (hist.dwell_frames > 5) { // Waiting for > 5 frames
                waiting_truck_count++;
                total_dwell_time += hist.dwell_frames;
            }
        }
        if (pimpl_->container_ids_.count(t.box.class_id) /* && is_in_zone(yard_zone)*/) {
            analytics.num_containers_stacked++;
        }

        // Visualization
        std::string label = pimpl_->labels_[t.box.class_id] + " #" + std::to_string(t.track_id);
        cv::rectangle(analytics.overview_image, box, cv::Scalar(255,0,0), 2);
        cv::putText(analytics.overview_image, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255), 2);
    }

    // Calculate KPIs
    analytics.num_trucks_waiting = waiting_truck_count;
    if (waiting_truck_count > 0 && fps > 0) {
        analytics.average_truck_wait_time_sec = (total_dwell_time / waiting_truck_count) / fps;
    }

    // Quay Occupancy: Sum of vessel areas in berth zone / total berth zone area
    // (Logic omitted for brevity)
    analytics.quay_occupancy_percent = 0.0f;

    return analytics;
}

} // namespace xinfer::zoo::maritime