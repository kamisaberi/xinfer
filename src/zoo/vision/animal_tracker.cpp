#include <xinfer/zoo/vision/animal_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct AnimalTracker::Impl {
    AnimalTrackerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_post_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;
    std::vector<std::string> labels_;

    Impl(const AnimalTrackerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("AnimalTracker: Failed to load model " + config_.model_path);
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
        detector_post_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        detector_post_->init(det_cfg);

        // 4. Setup Tracker (Kalman/SORT)
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = config_.max_track_age;
        trk_cfg.min_hits = config_.min_hits;
        // IoU threshold for matching detections to tracks
        trk_cfg.iou_threshold = 0.3f;
        tracker_->init(trk_cfg);

        // 5. Load Labels
        if (!config_.labels_path.empty()) {
            load_labels(config_.labels_path);
        }
    }

    void load_labels(const std::string& path) {
        std::ifstream file(path);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            labels_.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

AnimalTracker::AnimalTracker(const AnimalTrackerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

AnimalTracker::~AnimalTracker() = default;
AnimalTracker::AnimalTracker(AnimalTracker&&) noexcept = default;
AnimalTracker& AnimalTracker::operator=(AnimalTracker&&) noexcept = default;

void AnimalTracker::reset() {
    if (pimpl_ && pimpl_->tracker_) {
        pimpl_->tracker_->reset();
    }
}

std::vector<AnimalTrack> AnimalTracker::track(const cv::Mat& image) {
    if (!pimpl_) return {};

    // --- 1. Preprocess ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // --- 2. Inference ---
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // --- 3. Detect (Decode + NMS) ---
    auto raw_detections = pimpl_->detector_post_->process({pimpl_->output_tensor});

    // --- 4. Filtering & Scaling ---
    std::vector<postproc::BoundingBox> valid_detections;

    // Scale factors to map back to original image
    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (auto& det : raw_detections) {
        // Filter by class ID if a set is provided
        if (!pimpl_->config_.filter_class_ids.empty()) {
            if (pimpl_->config_.filter_class_ids.find(det.class_id) == pimpl_->config_.filter_class_ids.end()) {
                continue; // Skip this detection (not an animal we care about)
            }
        }

        // Scale coordinates now, so the Tracker works in Pixel Space
        det.x1 *= scale_x;
        det.y1 *= scale_y;
        det.x2 *= scale_x;
        det.y2 *= scale_y;

        valid_detections.push_back(det);
    }

    // --- 5. Update Tracker ---
    // The tracker associates these new detections with existing IDs
    auto tracked_objects = pimpl_->tracker_->update(valid_detections);

    // --- 6. Format Output ---
    std::vector<AnimalTrack> final_tracks;
    final_tracks.reserve(tracked_objects.size());

    for (const auto& obj : tracked_objects) {
        AnimalTrack t;
        t.track_id = obj.track_id;
        t.x1 = obj.box.x1;
        t.y1 = obj.box.y1;
        t.x2 = obj.box.x2;
        t.y2 = obj.box.y2;
        t.confidence = obj.box.confidence;
        t.velocity_x = obj.velocity_x;
        t.velocity_y = obj.velocity_y;

        // Label lookup
        if (obj.box.class_id >= 0 && obj.box.class_id < (int)pimpl_->labels_.size()) {
            t.species = pimpl_->labels_[obj.box.class_id];
        } else {
            t.species = "Unknown";
        }

        final_tracks.push_back(t);
    }

    return final_tracks;
}

} // namespace xinfer::zoo::vision