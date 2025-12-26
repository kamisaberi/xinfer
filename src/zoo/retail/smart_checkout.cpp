#include <xinfer/zoo/retail/smart_checkout.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>
#include <xinfer/postproc/vision/tracker_interface.h>

#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

namespace xinfer::zoo::retail {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SmartCheckout::Impl {
    CheckoutConfig config_;

    // --- Pipeline 1: Detection ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    core::Tensor det_input, det_output;

    // --- Pipeline 2: Tracking ---
    std::unique_ptr<postproc::ITracker> tracker_;
    // Keep track of IDs that have already been billed
    std::set<int> scanned_ids_;

    // --- Pipeline 3: Classification ---
    std::unique_ptr<backends::IBackend> cls_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> cls_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> cls_postproc_;
    core::Tensor cls_input, cls_output;

    // Database
    std::vector<std::string> sku_labels_;
    std::map<std::string, float> price_db_;

    // State
    CartSession current_cart_;
    std::vector<ScannedItem> current_frame_objects_; // For debug draw

    Impl(const CheckoutConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Setup Detector
        det_engine_ = backends::BackendFactory::create(config_.target);
        xinfer::Config d_cfg; d_cfg.model_path = config_.detector_model_path;
        d_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(d_cfg.model_path)) {
            throw std::runtime_error("SmartCheckout: Failed to load detector.");
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig dp_cfg;
        dp_cfg.target_width = config_.det_input_width;
        dp_cfg.target_height = config_.det_input_height;
        dp_cfg.target_format = preproc::ImageFormat::RGB;
        dp_cfg.layout_nchw = true;
        det_preproc_->init(dp_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig dpost_cfg;
        dpost_cfg.conf_threshold = config_.det_conf_thresh;
        // The detector just needs to find "Item", so 1 class is common
        // Or it could be "Bottle", "Box", "Fruit" (Coarse classes)
        det_postproc_->init(dpost_cfg);

        // 2. Setup Tracker
        tracker_ = postproc::create_tracker(config_.target);
        postproc::TrackerConfig trk_cfg;
        trk_cfg.max_age = 15; // Drop lost items quickly
        trk_cfg.min_hits = 3; // Confirm existence
        tracker_->init(trk_cfg);

        // 3. Setup Classifier
        if (!config_.classifier_model_path.empty()) {
            cls_engine_ = backends::BackendFactory::create(config_.target);
            xinfer::Config c_cfg; c_cfg.model_path = config_.classifier_model_path;

            if (cls_engine_->load_model(c_cfg.model_path)) {
                cls_preproc_ = preproc::create_image_preprocessor(config_.target);
                preproc::ImagePreprocConfig cp_cfg;
                cp_cfg.target_width = config_.cls_input_width;
                cp_cfg.target_height = config_.cls_input_height;
                cp_cfg.target_format = preproc::ImageFormat::RGB;
                cp_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}}; // ImageNet
                cls_preproc_->init(cp_cfg);

                cls_postproc_ = postproc::create_classification(config_.target);
                postproc::ClassificationConfig cpost_cfg;
                cpost_cfg.top_k = 1;
                load_db();
                cpost_cfg.labels = sku_labels_;
                cls_postproc_->init(cpost_cfg);
            }
        }

        reset_session();
    }

    void load_db() {
        // Load Labels
        std::ifstream file(config_.sku_labels_path);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                sku_labels_.push_back(line);
            }
        }

        // Load Prices (CSV: Name,Price)
        std::ifstream pfile(config_.price_db_path);
        if (pfile.is_open()) {
            std::string line;
            while (std::getline(pfile, line)) {
                std::stringstream ss(line);
                std::string name, price_s;
                if (std::getline(ss, name, ',') && std::getline(ss, price_s, ',')) {
                    try {
                        price_db_[name] = std::stof(price_s);
                    } catch(...) {}
                }
            }
        }
    }

    void reset_session() {
        current_cart_ = { {}, 0.0f, 0 };
        scanned_ids_.clear();
        tracker_->reset();
    }

    // Identify SKU from a crop
    std::pair<std::string, float> identify_sku(const cv::Mat& crop) {
        if (!cls_engine_) return {"Unknown", 0.0f};

        preproc::ImageFrame frame{crop.data, crop.cols, crop.rows, preproc::ImageFormat::BGR};
        cls_preproc_->process(frame, cls_input);
        cls_engine_->predict({cls_input}, {cls_output});
        auto results = cls_postproc_->process(cls_output);

        if (!results.empty() && !results[0].empty()) {
            return {results[0][0].label, results[0][0].score};
        }
        return {"Unknown", 0.0f};
    }
};

// =================================================================================
// Public API
// =================================================================================

SmartCheckout::SmartCheckout(const CheckoutConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SmartCheckout::~SmartCheckout() = default;
SmartCheckout::SmartCheckout(SmartCheckout&&) noexcept = default;
SmartCheckout& SmartCheckout::operator=(SmartCheckout&&) noexcept = default;

void SmartCheckout::new_session() {
    if (pimpl_) pimpl_->reset_session();
}

std::vector<ScannedItem> SmartCheckout::get_debug_objects() {
    if (!pimpl_) return {};
    return pimpl_->current_frame_objects_;
}

CartSession SmartCheckout::process_frame(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("SmartCheckout is null.");

    pimpl_->current_frame_objects_.clear();

    // 1. Detect
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});
    auto detections = pimpl_->det_postproc_->process({pimpl_->det_output});

    // Scale Boxes
    float sx = (float)image.cols / pimpl_->config_.det_input_width;
    float sy = (float)image.rows / pimpl_->config_.det_input_height;

    for (auto& d : detections) {
        d.x1 *= sx; d.x2 *= sx; d.y1 *= sy; d.y2 *= sy;
    }

    // 2. Track
    auto tracks = pimpl_->tracker_->update(detections);

    // 3. Logic: Scan Zone Check
    // Convert relative config rect to pixel rect
    cv::Rect scan_zone(
        (int)(pimpl_->config_.scan_zone.x * image.cols),
        (int)(pimpl_->config_.scan_zone.y * image.rows),
        (int)(pimpl_->config_.scan_zone.width * image.cols),
        (int)(pimpl_->config_.scan_zone.height * image.rows)
    );

    for (const auto& t : tracks) {
        // Current Object State
        ScannedItem obj;
        obj.track_id = t.track_id;
        obj.box = t.box;
        obj.sku_name = "Scanning...";
        obj.price = 0.0f;

        // Check if center is in zone
        float cx = (t.box.x1 + t.box.x2) / 2.0f;
        float cy = (t.box.y1 + t.box.y2) / 2.0f;
        cv::Point center((int)cx, (int)cy);

        if (scan_zone.contains(center)) {
            // Is this a new item?
            if (pimpl_->scanned_ids_.find(t.track_id) == pimpl_->scanned_ids_.end()) {

                // Crop and Classify
                cv::Rect roi((int)t.box.x1, (int)t.box.y1,
                             (int)(t.box.x2 - t.box.x1), (int)(t.box.y2 - t.box.y1));
                roi &= cv::Rect(0, 0, image.cols, image.rows);

                if (roi.width > 20 && roi.height > 20) {
                    cv::Mat crop = image(roi);
                    auto result = pimpl_->identify_sku(crop);

                    if (result.second > 0.6f) { // Confidence check
                        obj.sku_name = result.first;
                        obj.confidence = result.second;

                        // Add to Cart
                        pimpl_->scanned_ids_.insert(t.track_id);

                        if (pimpl_->price_db_.count(obj.sku_name)) {
                            obj.price = pimpl_->price_db_[obj.sku_name];
                        }

                        pimpl_->current_cart_.items.push_back(obj);
                        pimpl_->current_cart_.total_price += obj.price;
                        pimpl_->current_cart_.item_count++;

                        XINFER_LOG_INFO("Added to cart: " + obj.sku_name);
                    }
                }
            } else {
                obj.sku_name = "Added"; // Already processed
            }
        }

        pimpl_->current_frame_objects_.push_back(obj);
    }

    return pimpl_->current_cart_;
}

} // namespace xinfer::zoo::retail