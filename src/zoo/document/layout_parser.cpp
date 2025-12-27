#include <xinfer/zoo/document/layout_parser.h>
#include <xinfer/core/logging.h>

// --- Reuse the generic Object Detector ---
#include <xinfer/zoo/vision/detector.h>

#include <iostream>
#include <random>

namespace xinfer::zoo::document {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct LayoutParser::Impl {
    LayoutParserConfig config_;

    // High-level Zoo module for detection
    std::unique_ptr<vision::ObjectDetector> detector_;

    // Visualization colors
    std::map<std::string, cv::Scalar> color_map_;

    Impl(const LayoutParserConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Configure the Object Detector
        vision::DetectorConfig det_cfg;
        det_cfg.target = config_.target;
        det_cfg.model_path = config_.model_path;
        det_cfg.labels_path = config_.labels_path;
        det_cfg.input_width = config_.input_width;
        det_cfg.input_height = config_.input_height;
        det_cfg.confidence_threshold = config_.conf_threshold;
        det_cfg.nms_iou_threshold = config_.nms_threshold;
        det_cfg.vendor_params = config_.vendor_params;

        detector_ = std::make_unique<vision::ObjectDetector>(det_cfg);

        // 2. Setup visualization colors
        cv::RNG rng(12345);
        // Pre-defined colors for common elements
        color_map_["title"] = cv::Scalar(255, 0, 0); // Blue
        color_map_["paragraph"] = cv::Scalar(0, 255, 0); // Green
        color_map_["table"] = cv::Scalar(0, 0, 255); // Red
        color_map_["figure"] = cv::Scalar(255, 255, 0); // Cyan
        color_map_["list"] = cv::Scalar(255, 0, 255); // Magenta
    }

    cv::Scalar get_color(const std::string& label) {
        if (color_map_.count(label)) {
            return color_map_[label];
        }
        // Return a random color for unknown types
        return cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
    }
};

// =================================================================================
// Public API
// =================================================================================

LayoutParser::LayoutParser(const LayoutParserConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

LayoutParser::~LayoutParser() = default;
LayoutParser::LayoutParser(LayoutParser&&) noexcept = default;
LayoutParser& LayoutParser::operator=(LayoutParser&&) noexcept = default;

LayoutResult LayoutParser::parse(const cv::Mat& image) {