#include <xinfer/zoo/recycling/sorter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <iostream>
#include <fstream>
#include <algorithm>

namespace xinfer::zoo::recycling {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct Sorter::Impl {
    SorterConfig config_;

    // Pipeline Components
    std::unique_ptr<backends::IBackend> engine_;
    std::unique_ptr<preproc::IImagePreprocessor> preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> detector_post_;
    std::unique_ptr<postproc::ITracker> tracker_;

    // Data Containers
    core::Tensor input_tensor;
    core::Tensor output_tensor;

    // Labels
    std::vector<std::string> labels_;

    // Sorting Logic: Material -> Bin ID Mapping
    // This could be config driven, hardcoded for now
    std::map<std::string, int> material_to_bin_;

    Impl(const SorterConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Load Backend
        engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config backend_cfg;
        backend_cfg.model_path = config_.model_path;
        backend_cfg.vendor_params = config_.vendor_params;

        if (!engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("Sorter: Failed to load model " + config_.model_path);
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

        // 4. Setup Tracker (Critical for Conveyor)
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = 5;  // Items move fast, if lost for 5 frames, it's gone
        trk_cfg.min_hits = 2; // Confirm quickly
        tracker_->init(trk_cfg);

        // 5. Load Labels & Rules
        if (!config_.labels_path.empty()) load_labels(config_.labels_path);

        // Define Bin Rules (Example)
        material_to_bin_["Plastic"] = 1;
        material_to_bin_["PET"] = 1;
        material_to_bin_["Metal"] = 2;
        material_to_bin_["Aluminum"] = 2;
        material_to_bin_["Glass"] = 3;
        material_to_bin_["Paper"] = 4;
        // Everything else -> Bin 0 (Reject/Landfill)
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
};

// =================================================================================
// Public API
// =================================================================================

Sorter::Sorter(const SorterConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

Sorter::~Sorter() = default;
Sorter::Sorter(Sorter&&) noexcept = default;
Sorter& Sorter::operator=(Sorter&&) noexcept = default;

void Sorter::reset() {
    if (pimpl_ && pimpl_->tracker_) pimpl_->tracker_->reset();
}

std::vector<WasteItem> Sorter::process(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("Sorter is null.");

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
    auto raw_dets = pimpl_->detector_post_->process({pimpl_->output_tensor});

    // Scale Boxes to Image
    std::vector<postproc::BoundingBox> scaled_dets;
    float sx = (float)image.cols / pimpl_->config_.input_width;
    float sy = (float)image.rows / pimpl_->config_.input_height;

    for (auto& d : raw_dets) {
        d.x1 *= sx; d.x2 *= sx;
        d.y1 *= sy; d.y2 *= sy;
        scaled_dets.push_back(d);
    }

    // 4. Track
    auto tracks = pimpl_->tracker_->update(scaled_dets);

    // 5. Logic: Check Ejection Zones
    std::vector<WasteItem> items;
    items.reserve(tracks.size());

    for (const auto& t : tracks) {
        WasteItem item;
        item.track_id = t.track_id;
        item.box = t.box;
        item.confidence = t.box.confidence;

        // Calculate Center (Grasp Point)
        item.center.x = (t.box.x1 + t.box.x2) / 2.0f;
        item.center.y = (t.box.y1 + t.box.y2) / 2.0f;

        // Label
        if (t.box.class_id >= 0 && t.box.class_id < (int)pimpl_->labels_.size()) {
            item.material = pimpl_->labels_[t.box.class_id];
        } else {
            item.material = "Unknown";
        }

        // Determine Bin
        if (pimpl_->material_to_bin_.count(item.material)) {
            item.target_bin_id = pimpl_->material_to_bin_[item.material];
        } else {
            item.target_bin_id = 0; // Default
        }

        // Check Trigger Line
        // Assumes conveyor moves along Y axis (Top to Bottom)
        item.in_ejection_zone = false;

        if (pimpl_->config_.ejection_lines.count(item.target_bin_id)) {
            int line_y = pimpl_->config_.ejection_lines[item.target_bin_id];
            int tol = pimpl_->config_.ejection_tolerance_px;

            // Check if center is crossing the line
            if (item.center.y >= (line_y - tol) && item.center.y <= (line_y + tol)) {
                item.in_ejection_zone = true;
            }
        }

        items.push_back(item);
    }

    return items;
}

} // namespace xinfer::zoo::recycling