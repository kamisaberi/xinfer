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

    /**
     * @brief A recognized product item.
     */
    struct ScannedItem {
        int track_id;          // Tracker ID to prevent double scanning
        std::string sku_name;  // "Coke_330ml"
        float price;           // Derived from database
        float confidence;
        postproc::BoundingBox box;
    };

    /**
     * @brief Current session state (the "Virtual Cart").
     */
    struct CartSession {
        std::vector<ScannedItem> items;
        float total_price;
        int item_count;
    };

    struct CheckoutConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Product Detector (Locates items) ---
        std::string detector_model_path;
        int det_input_width = 640;
        int det_input_height = 640;
        float det_conf_thresh = 0.5f;

        // --- Model 2: SKU Classifier (Identifies items) ---
        std::string classifier_model_path;
        std::string sku_labels_path;   // List of SKU names
        std::string price_db_path;     // CSV: Name,Price
        int cls_input_width = 224;
        int cls_input_height = 224;

        // --- Logic ---
        // Define a "Scan Zone" (ROI). Items are added to cart when their center enters this box.
        // Normalized coordinates [0.0 - 1.0]
        cv::Rect2f scan_zone = {0.2f, 0.2f, 0.6f, 0.6f};

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class SmartCheckout {
    public:
        explicit SmartCheckout(const CheckoutConfig& config);
        ~SmartCheckout();

        // Move semantics
        SmartCheckout(SmartCheckout&&) noexcept;
        SmartCheckout& operator=(SmartCheckout&&) noexcept;
        SmartCheckout(const SmartCheckout&) = delete;
        SmartCheckout& operator=(const SmartCheckout&) = delete;

        /**
         * @brief Process a video frame from the checkout camera.
         *
         * Logic:
         * 1. Detect objects.
         * 2. Track objects.
         * 3. If a tracked object enters the "Scan Zone" and hasn't been scanned:
         *    a. Crop object.
         *    b. Classify SKU.
         *    c. Add to Cart.
         *
         * @param image Input frame.
         * @return The updated Cart Session.
         */
        CartSession process_frame(const cv::Mat& image);

        /**
         * @brief Manually reset the cart (e.g. new customer).
         */
        void new_session();

        /**
         * @brief Get debug info (current tracked bounding boxes and IDs).
         * Useful for drawing the UI.
         */
        std::vector<ScannedItem> get_debug_objects();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail