#pragma once


#include <string>
#include <vector>
#include <memory>
#include <map>
#include <opencv2/opencv.hpp>
#include <include/zoo/vision/detector.h>

namespace xinfer::zoo::retail {

    struct CartItem {
        int class_id;
        std::string label;
        int quantity;
        float total_confidence;
    };

    struct SmartCheckoutConfig {
        vision::DetectorConfig detector_config;
        float tracking_iou_threshold = 0.5f;
    };

    class SmartCheckout {
    public:
        explicit SmartCheckout(const SmartCheckoutConfig& config);
        ~SmartCheckout();

        SmartCheckout(const SmartCheckout&) = delete;
        SmartCheckout& operator=(const SmartCheckout&) = delete;
        SmartCheckout(SmartCheckout&&) noexcept;
        SmartCheckout& operator=(SmartCheckout&&) noexcept;

        void process_frame(int camera_id, const cv::Mat& frame);

        std::vector<CartItem> get_customer_cart(int customer_id);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail

