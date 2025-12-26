#include <xinfer/zoo/geospatial/maritime_detector.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <fstream>
#include <iostream>

namespace xinfer::zoo::geospatial {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct MaritimeDetector::Impl {
    MaritimeConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Labels
    std::vector<std::string> labels_;

    Impl(const MaritimeConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("MaritimeDetector: Failed to load model " + config_.model_path);
        }

        // 2. Setup Preprocessor
        preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig pre_cfg;
        pre_cfg.target_width = config_.input_width;
        pre_cfg.target_height = config_.input_height;
        preproc_->init(pre_cfg);

        // 3. Setup Detector Post-processor
        detector_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_cfg;
        det_cfg.conf_threshold = config_.conf_threshold;
        det_cfg.nms_threshold = config_.nms_threshold;
        load_labels(config_.labels_path);
        det_cfg.num_classes = labels_.size();
        detector_->init(det_cfg);

        // 4. Setup Tracker (Optional)
        if (config_.enable_tracking) {
            tracker_ = postproc::create_tracker(config_.target);
            postproc::TrackerConfig trk_cfg;
            tracker_->init(trk_cfg);
        }
    }

    void load_labels(const std::string& path) {
        if (path.empty()) return;
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

MaritimeDetector::MaritimeDetector(const MaritimeConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

MaritimeDetector::~MaritimeDetector() = default;
MaritimeDetector::MaritimeDetector(MaritimeDetector&&) noexcept = default;
MaritimeDetector& MaritimeDetector::operator=(MaritimeDetector&&) noexcept = default;

void MaritimeDetector::reset_tracker() {
    if (pimpl_ && pimpl_->tracker_) pimpl_->tracker_->reset();
}

std::vector<Vessel> MaritimeDetector::detect(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("MaritimeDetector is null.");

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
    auto detections = pimpl_->detector_->process({pimpl_->output_tensor});

    // Scale boxes
    float sx = (float)image.cols / pimpl_->config_.input_width;
    float sy = (float)image.rows / pimpl_->config_.input_height;

    for (auto& d : detections) {
        d.x1 *= sx; d.x2 *= sx; d.y1 *= sy; d.y2 *= sy;
    }

    std::vector<Vessel> results;

    // 4. Track or Direct Output
    if (pimpl_->tracker_) {
        auto tracks = pimpl_->tracker_->update(detections);

        results.reserve(tracks.size());
        for (const auto& t : tracks) {
            Vessel v;
            v.track_id = t.track_id;
            v.box = t.box;
            v.confidence = t.box.confidence;

            if (t.box.class_id < (int)pimpl_->labels_.size()) {
                v.type = pimpl_->labels_[t.box.class_id];
            }
            results.push_back(v);
        }
    } else {
        // No tracking, just output detections
        results.reserve(detections.size());
        for (const auto& d : detections) {
            Vessel v;
            v.track_id = -1; // No tracking ID
            v.box = d;
            v.confidence = d.confidence;

            if (d.class_id < (int)pimpl_->labels_.size()) {
                v.type = pimpl_->labels_[d.class_id];
            }
            results.push_back(v);
        }
    }

    return results;
}

} // namespace xinfer::zoo::geospatial