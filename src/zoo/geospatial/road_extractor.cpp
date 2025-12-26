#include <xinfer/zoo/geospatial/road_extractor.h>
#include <xinfer/core/logging.h>
#include <xinfer/core/tensor.h>

// --- We reuse the Segmenter Zoo module for the core AI logic ---
#include <xinfer/zoo/vision/segmenter.h>

#include <iostream>
#include <numeric>

namespace xinfer::zoo::geospatial {

// =================================================================================
// PImpl Implementation
// =================================================================================

struct RoadExtractor::Impl {
    RoadConfig config_;

    // High-level Zoo module for segmentation
    std::unique_ptr<vision::Segmenter> segmenter_;

    Impl(const RoadConfig& config) : config_(config) {
        initialize();
    }

    void initialize() {
        // 1. Initialize the Segmenter
        vision::SegmenterConfig seg_cfg;
        seg_cfg.target = config_.target;
        seg_cfg.model_path = config_.model_path;
        seg_cfg.input_width = config_.input_width;
        seg_cfg.input_height = config_.input_height;
        seg_cfg.vendor_params = config_.vendor_params;

        segmenter_ = std::make_unique<vision::Segmenter>(seg_cfg);
    }

    // --- Core Logic: Find Lane Centerline ---
    void find_centerline(const cv::Mat& mask, RoadResult& result) {
        // Iterate over horizontal slices of the mask from bottom to top
        for (int y = mask.rows - 1; y > mask.rows * 0.5; y -= 10) { // Scan bottom half

            // Find start and end of road segment on this row
            const uint8_t* row = mask.ptr<uint8_t>(y);
            int first = -1, last = -1;

            for (int x = 0; x < mask.cols; ++x) {
                if (row[x] > 0) {
                    if (first == -1) first = x;
                    last = x;
                }
            }

            // If a valid road segment is found, add its center to the path
            if (first != -1 && (last - first) > 10) { // Min width
                result.lane_centerline.emplace_back((float)(first + last) / 2.0f, (float)y);
            }
        }
    }
};

// =================================================================================
// Public API
// =================================================================================

RoadExtractor::RoadExtractor(const RoadConfig& config)
    : pimpl_(std::make_unique<Impl>(config)) {}

RoadExtractor::~RoadExtractor() = default;
RoadExtractor::RoadExtractor(RoadExtractor&&) noexcept = default;
RoadExtractor& RoadExtractor::operator=(RoadExtractor&&) noexcept = default;

RoadResult RoadExtractor::extract(const cv::Mat& image) {
    if (!pimpl_) throw std::runtime_error("RoadExtractor is null.");

    // 1. Run Segmentation
    auto seg_res = pimpl_->segmenter_->segment(image);

    // 2. Extract Road Mask
    // Seg res contains a class mask. We need to isolate the "Road" class.
    cv::Mat road_mask;
    cv::inRange(seg_res.mask, cv::Scalar(pimpl_->config_.road_class_id),
                cv::Scalar(pimpl_->config_.road_class_id), road_mask);

    RoadResult result;
    result.road_mask = road_mask;

    // 3. Quantification
    float road_pixels = cv::countNonZero(road_mask);
    result.drivable_area_sqm = road_pixels * pimpl_->config_.sq_meters_per_pixel;

    // 4. Find Path
    pimpl_->find_centerline(road_mask, result);

    // 5. Visualization
    // Blend original image with a colored road mask
    cv::Mat color_layer = cv::Mat::zeros(image.size(), CV_8UC3);
    color_layer.setTo(cv::Scalar(0, 255, 0), road_mask); // Green for road
    cv::addWeighted(image, 1.0, color_layer, 0.3, 0.0, result.overlay);

    // Draw centerline
    for (size_t i = 1; i < result.lane_centerline.size(); ++i) {
        cv::line(result.overlay, result.lane_centerline[i-1], result.lane_centerline[i],
                 cv::Scalar(255, 255, 0), 2);
    }

    return result;
}

} // namespace xinfer::zoo::geospatial