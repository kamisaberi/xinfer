#include <xinfer/zoo/space/debris_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>

namespace xinfer::zoo::space {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct DebrisTracker::Impl {
    DebrisConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const DebrisConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("DebrisTracker: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        // Space cameras are often monochrome (Star Trackers)
        // If model was trained on RGB, preproc handles conversion.
        // If trained on Gray, change this to GRAY.
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Detector (Small Object Detection)
        detector_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        detector_->init(det_cfg);

        // 4. Setup Tracker (Kalman Filter)
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = config_.max_track_age;
        trk_cfg.min_hits = config_.min_hits;
        // In space, objects don't overlap much, so IoU matching can be loose
        trk_cfg.iou_threshold = 0.1f;
        tracker_->init(trk_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

DebrisTracker::DebrisTracker(const DebrisConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

DebrisTracker::~DebrisTracker() = default;
DebrisTracker::DebrisTracker(DebrisTracker&&) noexcept = default;
DebrisTracker& DebrisTracker::operator=(DebrisTracker&&) noexcept = default;

void DebrisTracker::reset() {
    if (pimpl_ && pimpl_->tracker_) {
        pimpl_->tracker_->reset();
    }
}

std::vector<DebrisObject> DebrisTracker::track(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("DebrisTracker is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    // Handle input format (OpenCV reads as BGR or Gray)
    frame.format = (image.channels() == 1) ? preproc::ImageFormat::GRAY : preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect (Decode + NMS)
    auto raw_detections = pimpl_->detector_->process({pimpl_->output_tensor});

    // 4. Scale to original image
    std::vector<postproc::BoundingBox> scaled_dets;
    scaled_dets.reserve(raw_detections.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : raw_detections) {
        postproc::BoundingBox b = det;
        b.x1 *= scale_x; b.x2 *= scale_x;
        b.y1 *= scale_y; b.y2 *= scale_y;
        scaled_dets.push_back(b);
    }

    // 5. Update Tracker
    // This is the critical step for Space:
    // It filters out random sensor noise (hot pixels) because they don't move
    // in a predictable Kalman trajectory, so they won't get 'confirmed' status.
    auto tracks = pimpl_->tracker_->update(scaled_dets);

    // 6. Output
    std::vector<DebrisObject> results;
    results.reserve(tracks.size());

    for (const auto& t : tracks) {
        DebrisObject obj;
        obj.track_id = t.track_id;

        // Calculate center and size
        obj.w = t.box.x2 - t.box.x1;
        obj.h = t.box.y2 - t.box.y1;
        obj.x = t.box.x1 + obj.w * 0.5f;
        obj.y = t.box.y1 + obj.h * 0.5f;

        obj.confidence = t.box.confidence;
        obj.velocity_x = t.velocity_x;
        obj.velocity_y = t.velocity_y;

        results.push_back(obj);
    }

    return results;
}

} // namespace xinfer::zoo::space