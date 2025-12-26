#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::retail {

    struct ProductItem {
        std::string sku_name;
        int class_id;
        float confidence;
        postproc::BoundingBox box;
    };

    struct ShelfState {
        // List of all detected items
        std::vector<ProductItem> items;

        // List of detected empty spaces (if model supports "Gap" class)
        std::vector<postproc::BoundingBox> gaps;

        // Aggregated Inventory count (SKU -> Count)
        std::map<std::string, int> inventory_summary;
    };

    struct AuditorConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // Model Path (e.g., yolov8_shelf.rknn)
        std::string model_path;

        // Path to SKU names file (coco.names style)
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Settings
        float conf_threshold = 0.4f;
        float nms_threshold = 0.5f;

        // Optional: specific class ID representing "Empty/Gap"
        // Set to -1 if model doesn't detect gaps.
        int gap_class_id = -1;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class ShelfAuditor {
    public:
        explicit ShelfAuditor(const AuditorConfig& config);
        ~ShelfAuditor();

        // Move semantics
        ShelfAuditor(ShelfAuditor&&) noexcept;
        ShelfAuditor& operator=(ShelfAuditor&&) noexcept;
        ShelfAuditor(const ShelfAuditor&) = delete;
        ShelfAuditor& operator=(const ShelfAuditor&) = delete;

        /**
         * @brief Scan a shelf image for inventory.
         *
         * @param image Input frame (High res recommended for small SKUs).
         * @return ShelfState containing item locations and counts.
         */
        ShelfState audit(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail