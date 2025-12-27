#include <xinfer/zoo/energy/seismic_interpreter.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We reuse the generic Segmenter module ---
#include <xinfer/zoo/vision/segmenter.h>

#include <iostream>
#include <vector>

namespace xinfer::zoo::energy {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct SeismicInterpreter::Impl {
    SeismicConfig config_;

    // High-level Zoo module for segmentation
    std::unique_ptr<vision::Segmenter> segmenter_;

    Impl(const SeismicConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Initialize the Segmenter
        vision::SegmenterConfig seg_cfg;
        seg_cfg.target = config_.target;
        seg_cfg.model_path = config_.model_path;
        seg_cfg.input_width = config_.input_width;
        seg_cfg.input_height = config_.input_height;
        seg_cfg.class_names = config_.class_names;
        seg_cfg.class_colors = config_.class_colors;
        seg_cfg.vendor_params = config_.vendor_params;

        segmenter_ = std::make_unique<vision::Segmenter>(seg_cfg);
    }

    // --- Core Logic: Mask -> Contours ---
    std::vector<GeoFeature> extract_features(const cv::Mat& mask) {
        std::vector<GeoFeature> features;
        int num_classes = config_.class_names.size();

        for (int c = 1; c < num_classes; ++c) { // Skip background (class 0)
            cv::Mat class_mask;
            cv::inRange(mask, cv::Scalar(c), cv::Scalar(c), class_mask);

            if (cv::countNonZero(class_mask) > 0) {
                // Find contours for this class
                std::vector<std::vector<cv::Point>> contours;
                cv::findContours(class_mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

                for (const auto& cnt : contours) {
                    // Filter small noise
                    if (cv::contourArea(cnt) > 20) {
                        GeoFeature feat;
                        feat.type = config_.class_names[c];
                        feat.confidence = 0.9f; // Placeholder
                        feat.contour = cnt;
                        features.push_back(feat);
                    }
                }
            }
        }
        return features;
    }
};

// =================================================================================
// Public API
// =================================================================================

SeismicInterpreter::SeismicInterpreter(const SeismicConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

SeismicInterpreter::~SeismicInterpreter() = default;
SeismicInterpreter::SeismicInterpreter(SeismicInterpreter&&) noexcept = default;
SeismicInterpreter& SeismicInterpreter::operator=(SeismicInterpreter&&) noexcept = default;

InterpretationResult SeismicInterpreter::interpret_slice(const cv::Mat& seismic_slice) {
    if (!pimpl_ || !pimpl_->segmenter_) throw std::runtime_error("SeismicInterpreter is null.");

    // 1. Run Segmentation
    // Input is usually single-channel float, but we pass it as a Mat
    // The Segmenter's preprocessor will handle normalization and channel conversion.
    auto seg_res = pimpl_->segmenter_->segment(seismic_slice);

    InterpretationResult result;
    result.geo_map = seg_res.color_mask;

    // 2. Extract Geometric Features
    result.features = pimpl_->extract_features(seg_res.mask);

    // 3. Draw contours on visualization
    for (const auto& feat : result.features) {
        // Draw a bounding box for the feature
        cv::Rect box = cv::boundingRect(feat.contour);
        cv::rectangle(result.geo_map, box, cv::Scalar(255, 255, 255), 1);
        cv::putText(result.geo_map, feat.type, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255,255,255));
    }

    return result;
}

} // namespace xinfer::zoo::energy