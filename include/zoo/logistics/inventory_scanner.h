#pragma once

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::logistics {

    struct ScannedItem {
        std::string sku_name; // Product Name / ID
        int class_id;
        float confidence;
        postproc::BoundingBox box;
    };

    struct InventoryReport {
        // List of all items found in the image
        std::vector<ScannedItem> items;

        // Tally of items (SKU -> Count)
        std::map<std::string, int> summary;

        // Timestamp of the scan
        long long timestamp;

        // Visualization
        cv::Mat annotated_image;
    };

    struct ScannerConfig {
        // Hardware Target (Edge CPU/NPU on a handheld or drone)
        xinfer::Target target = xinfer::Target::ROCKCHIP_RKNN;

        // Model Path (e.g., yolov8_sku_detector.rknn)
        std::string model_path;

        // Label Map (Class ID -> SKU Name)
        std::string labels_path;

        // Input Specs
        int input_width = 640;
        int input_height = 640;

        // Detection Sensitivity
        float conf_threshold = 0.6f; // Higher threshold to reduce false positives
        float nms_threshold = 0.5f;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class InventoryScanner {
    public:
        explicit InventoryScanner(const ScannerConfig& config);
        ~InventoryScanner();

        // Move semantics
        InventoryScanner(InventoryScanner&&) noexcept;
        InventoryScanner& operator=(InventoryScanner&&) noexcept;
        InventoryScanner(const InventoryScanner&) = delete;
        InventoryScanner& operator=(const InventoryScanner&) = delete;

        /**
         * @brief Scan an image of a shelf or pallet.
         *
         * @param image Input photo.
         * @return Inventory report with counts and locations.
         */
        InventoryReport scan(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::logistics