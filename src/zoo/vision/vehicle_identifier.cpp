#include <xinfer/zoo/vision/vehicle_identifier.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- The Three Pillars ---
#include <xinfer/backends/backend_factory.h>
#include <xinfer/preproc/factory.h>
#include <xinfer/postproc/factory.h>
#include <xinfer/postproc/vision/classification_interface.h>

#include <fstream>
#include <iostream>
#include <set>

namespace xinfer::zoo::vision {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct VehicleIdentifier::Impl {
    VehicleIdentifierConfig config_;

    // --- Stage 1: Detector ---
    std::unique_ptr<backends::IBackend> det_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> det_preproc_;
    std::unique_ptr<postproc::IDetectionPostprocessor> det_postproc_;
    core::Tensor det_input, det_output;
    std::vector<std::string> det_labels_;
    std::set<int> vehicle_class_ids_; // IDs for Car, Truck, Bus, Bike

    // --- Stage 2: Classifier ---
    std::unique_ptr<backends::IBackend> attr_engine_;
    std::unique_ptr<preproc::IImagePreprocessor> attr_preproc_;
    std::unique_ptr<postproc::IClassificationPostprocessor> attr_postproc_;
    core::Tensor attr_input, attr_output;
    std::vector<std::string> attr_labels_;

    Impl(const VehicleIdentifierConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // ---------------------------------------------------------
        // 1. Setup Detector
        // ---------------------------------------------------------
        det_engine_ = backends::BackendFactory::create(config_.target);

        xinfer::Config det_backend_cfg;
        det_backend_cfg.model_path = config_.det_model_path;
        det_backend_cfg.vendor_params = config_.vendor_params;

        if (!det_engine_->load_model(backend_cfg.model_path)) {
            throw std::runtime_error("VehicleIdentifier: Failed to load detector " + config_.det_model_path);
        }

        det_preproc_ = preproc::create_image_preprocessor(config_.target);
        preproc::ImagePreprocConfig det_pre_cfg;
        det_pre_cfg.target_width = config_.det_input_width;
        det_pre_cfg.target_height = config_.det_input_height;
        det_pre_cfg.target_format = preproc::ImageFormat::RGB;
        det_pre_cfg.layout_nchw = true;
        det_preproc_->init(det_pre_cfg);

        det_postproc_ = postproc::create_detection(config_.target);
        postproc::DetectionConfig det_post_cfg;
        det_post_cfg.conf_threshold = config_.det_conf_thresh;
        det_post_cfg.nms_threshold = config_.det_nms_thresh;
        det_postproc_->init(det_post_cfg);

        load_labels(config_.det_labels_path, det_labels_);

        // Identify which IDs are vehicles (Simple heuristic based on standard COCO names)
        for (size_t i = 0; i < det_labels_.size(); ++i) {
            std::string l = det_labels_[i];
            // Normalize string to lower case if needed
            if (l == "car" || l == "truck" || l == "bus" || l == "motorcycle") {
                vehicle_class_ids_.insert(i);
            }
        }
        // If no labels provided, assume standard COCO indices: 2, 5, 7, 3
        if (det_labels_.empty()) {
            vehicle_class_ids_ = {2, 3, 5, 7};
        }

        // ---------------------------------------------------------
        // 2. Setup Attribute Classifier (Optional)
        // ---------------------------------------------------------
        if (!config_.attr_model_path.empty()) {
            attr_engine_ = backends::BackendFactory::create(config_.target);
            xinfer::Config attr_backend_cfg;
            attr_backend_cfg.model_path = config_.attr_model_path;

            if (!attr_engine_->load_model(attr_backend_cfg.model_path)) {
                XINFER_LOG_WARN("Failed to load attribute model. Running detection only.");
                return;
            }

            attr_preproc_ = preproc::create_image_preprocessor(config_.target);
            preproc::ImagePreprocConfig attr_pre_cfg;
            attr_pre_cfg.target_width = config_.attr_input_width;
            attr_pre_cfg.target_height = config_.attr_input_height;
            attr_pre_cfg.target_format = preproc::ImageFormat::RGB;
            // Standard ImageNet normalization for classifiers
            attr_pre_cfg.norm_params = {{123.675, 116.28, 103.53}, {58.395, 57.12, 57.375}};
            attr_preproc_->init(attr_pre_cfg);

            attr_postproc_ = postproc::create_classification(config_.target);
            postproc::ClassificationConfig cls_cfg;
            cls_cfg.top_k = 1;
            load_labels(config_.attr_labels_path, attr_labels_);
            cls_cfg.labels = attr_labels_;
            attr_postproc_->init(cls_cfg);
        }
    }

    void load_labels(const std::string& path, std::vector<std::string>& list) {
        if (path.empty()) return;
        std::ifstream file(path);
        if (!file.is_open()) return;
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            list.push_back(line);
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

VehicleIdentifier::VehicleIdentifier(const VehicleIdentifierConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

VehicleIdentifier::~VehicleIdentifier() = default;
VehicleIdentifier::VehicleIdentifier(VehicleIdentifier&&) noexcept = default;
VehicleIdentifier& VehicleIdentifier::operator=(VehicleIdentifier&&) noexcept = default;

std::vector<VehicleResult> VehicleIdentifier::identify(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("VehicleIdentifier is null.");

    std::vector<VehicleResult> results;

    // --- 1. Detection ---
    preproc::ImageFrame frame;
    frame.data = image.data;
    frame.width = image.cols;
    frame.height = image.rows;
    frame.format = preproc::ImageFormat::BGR;

    pimpl_->det_preproc_->process(frame, pimpl_->det_input);
    pimpl_->det_engine_->predict({pimpl_->det_input}, {pimpl_->det_output});

    auto detections = pimpl_->det_postproc_->process({pimpl_->det_output});

    // Scale factors
    float scale_x = (float)image.cols / pimpl_->config_.det_input_width;
    float scale_y = (float)image.rows / pimpl_->config_.det_input_height;

    for (const auto& det : detections) {
        // Filter: Is this a vehicle?
        if (pimpl_->vehicle_class_ids_.find(det.class_id) == pimpl_->vehicle_class_ids_.end()) {
            if (!pimpl_->det_labels_.empty()) continue; // Skip if labels known and not vehicle
        }

        VehicleResult res;

        // Scale Box
        res.box.x1 = det.x1 * scale_x;
        res.box.y1 = det.y1 * scale_y;
        res.box.x2 = det.x2 * scale_x;
        res.box.y2 = det.y2 * scale_y;
        res.box.confidence = det.confidence;

        if (det.class_id < (int)pimpl_->det_labels_.size()) {
            res.type = pimpl_->det_labels_[det.class_id];
        } else {
            res.type = "Vehicle";
        }

        // --- 2. Attribute Classification (Optional) ---
        if (pimpl_->attr_engine_) {
            // Crop Vehicle
            cv::Rect roi(
                (int)std::max(0.0f, res.box.x1),
                (int)std::max(0.0f, res.box.y1),
                (int)(res.box.x2 - res.box.x1),
                (int)(res.box.y2 - res.box.y1)
            );

            // Bounds Check
            roi &= cv::Rect(0, 0, image.cols, image.rows);

            if (roi.width > 10 && roi.height > 10) {
                cv::Mat crop = image(roi);

                // Preprocess Crop
                preproc::ImageFrame crop_frame{crop.data, crop.cols, crop.rows, preproc::ImageFormat::BGR};
                pimpl_->attr_preproc_->process(crop_frame, pimpl_->attr_input);

                // Inference
                pimpl_->attr_engine_->predict({pimpl_->attr_input}, {pimpl_->attr_output});

                // Postprocess
                auto attrs = pimpl_->attr_postproc_->process(pimpl_->attr_output);

                if (!attrs.empty() && !attrs[0].empty()) {
                    res.make_model = attrs[0][0].label;
                    res.attr_confidence = attrs[0][0].score;
                    // Note: Color might need a secondary model or a multi-head output parsing
                    // here we assume the classifier returns Make/Model.
                }
            }
        }

        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::vision