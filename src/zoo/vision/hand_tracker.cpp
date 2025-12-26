#include <xinfer/zoo/vision/hand_tracker.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>

#include <iostream>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct HandTracker::Impl {
    HandTrackerConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_post_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    Impl(const HandTrackerConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("HandTracker: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        pre_cfg.target_format = preproc::ImageFormat::RGB;
        pre_cfg.layout_nchw = true;
        preproc_->init(pre_cfg);

        // 3. Setup Detector (YOLO Postproc)
        detector_post_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        // Hand detectors usually have 1 class ("Hand")
        det_cfg.num_classes = 1;
        detector_post_->init(det_cfg);

        // 4. Setup Tracker
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = config_.max_track_age;
        trk_cfg.min_hits = config_.min_hits;
        // Hands can move very fast, so we might need a looser IoU threshold
        trk_cfg.iou_threshold = 0.2f;
        tracker_->init(trk_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

HandTracker::HandTracker(const HandTrackerConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

HandTracker::~HandTracker() = default;
HandTracker::HandTracker(HandTracker&&) noexcept = default;
HandTracker& HandTracker::operator=(HandTracker&&) noexcept = default;

void HandTracker::reset() {
    if (pimpl_ && pimpl_->tracker_) {
        pimpl_->tracker_->reset();
    }
}

std::vector<HandResult> HandTracker::track(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("HandTracker is null.");

    // 1. Preprocess
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->preproc_->process(frame, pimpl_->input_tensor);

    // 2. Inference
    pimpl_->engine_->predict({pimpl_->input_tensor}, {pimpl_->output_tensor});

    // 3. Detect (Get Boxes)
    auto raw_detections = pimpl_->detector_post_->process({pimpl_->output_tensor});

    // 4. Scale Boxes back to Image Size
    std::vector<postproc::BoundingBox> valid_detections;
    valid_detections.reserve(raw_detections.size());

    float scale_x = (float)image.cols / pimpl_->config_.input_width;
    float scale_y = (float)image.rows / pimpl_->config_.input_height;

    for (const auto& det : raw_detections) {
        postproc::BoundingBox scaled_det = det;
        scaled_det.x1 *= scale_x;
        scaled_det.y1 *= scale_y;
        scaled_det.x2 *= scale_x;
        scaled_det.y2 *= scale_y;
        valid_detections.push_back(scaled_det);
    }

    // 5. Update Tracker
    auto tracks = pimpl_->tracker_->update(valid_detections);

    // 6. Map to Result
    std::vector<HandResult> results;
    results.reserve(tracks.size());

    for (const auto& t : tracks) {
        HandResult res;
        res.track_id = t.track_id;
        res.box = t.box;
        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::vision