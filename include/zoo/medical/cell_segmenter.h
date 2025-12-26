#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::medical {

    /**
     * @brief A single segmented cell.
     */
    struct CellObject {
        int id;
        postproc::BoundingBox box;
        cv::Mat mask;           // Binary mask for this specific cell

        // Morphological Metrics
        float area_pixels;      // Raw pixel count
        float area_microns;     // Calibrated size
        float circularity;      // 1.0 = perfect circle, < 0.5 = irregular
        float mean_intensity;   // Brightness (for fluorescence markers)
    };

    struct CellResult {
        std::vector<CellObject> cells;
        int total_count;
        float average_size_microns;
        cv::Mat segmentation_overlay; // Visual debug image
    };

    struct CellConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., cellpose_yolo.engine)
        std::string model_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Calibration
        float microns_per_pixel = 0.5f;

        // Thresholds
        float conf_threshold = 0.5f;
        float nms_threshold = 0.4f;
        float min_area_px = 20.0f; // Ignore noise/debris

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CellSegmenter {
    public:
        explicit CellSegmenter(const CellConfig& config);
        ~CellSegmenter();

        // Move semantics
        CellSegmenter(CellSegmenter&&) noexcept;
        CellSegmenter& operator=(CellSegmenter&&) noexcept;
        CellSegmenter(const CellSegmenter&) = delete;
        CellSegmenter& operator=(const CellSegmenter&) = delete;

        /**
         * @brief Segment and analyze cells in a microscopy image.
         *
         * Pipeline:
         * 1. Preprocess (Normalization).
         * 2. Inference (Instance Segmentation).
         * 3. Postprocess (Mask decoding).
         * 4. Morphometrics (Area, Circularity calc).
         *
         * @param image Input image (Gray or RGB).
         * @return Detection results and stats.
         */
        CellResult segment(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::medical