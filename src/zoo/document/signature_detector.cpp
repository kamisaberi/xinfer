#include <xinfer/zoo/document/signature_detector.h>
#include <xinfer/core/logging.h>

// --- Reuse the generic Object Detector ---
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <algorithm>

namespace xinfer::zoo::document {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SignatureDetector::Impl {
    SignatureConfig config_;

    // High-level Zoo module for detection
    std::unique_ptr<vision::ObjectDetector> detector_;

    Impl(const SignatureConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // Configure the Object Detector for this specific task
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.model_path;

        // We assume the model has only one class: "signature"
        // The labels_path is often not needed if we know the class index is 0.

        det_cfg.input_width = config_.input_width;
        det_cfg.input_height = config_.input_height;
        det_cfg.confidence_threshold = config_.conf_threshold;
        det_cfg.nms_iou_threshold = 0.4f; // Signatures shouldn't overlap much
        det_cfg.vendor_params = config_.vendor_params;

        detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);
    }
};

// =================================================================================
// Public API
// =================================================================================

SignatureDetector::SignatureDetector(const SignatureConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SignatureDetector::~SignatureDetector() = default;
SignatureDetector::SignatureDetector(SignatureDetector&&) noexcept = default;
SignatureDetector& SignatureDetector::operator=(SignatureDetector&&) noexcept = default;

std::vector<SignatureResult> SignatureDetector::detect(const cv::Mat& image) {
    if (!pimpl_ || !pimpl_->detector_) {
        throw std::runtime_error("SignatureDetector is not initialized.");
    }

    std::vector<SignatureResult> results;

    // 1. Run Detection
    auto detections = pimpl_->detector_->predict(image);

    // 2. Map to SignatureResult
    for (const auto& det : detections) {
        // If the model is multi-class, we might need to filter for class_id == "signature"
        // Here, we assume a single-class model where all detections are signatures.

        SignatureResult res;
        res.signature_found = true;
        res.confidence = det.confidence;
        res.box = {det.x1, det.y1, det.x2, det.y2, det.confidence, det.class_id};

        results.push_back(res);
    }

    // If no detections found, we can return a single "not found" result
    if (results.empty()) {
        SignatureResult res;
        res.signature_found = false;
        res.confidence = 0.0f;
        results.push_back(res);
    }

    return results;
}

} // namespace xinfer::zoo::document