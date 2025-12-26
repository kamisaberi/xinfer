#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::retail {

    struct CustomerAttributes {
        std::string gender;    // "Male", "Female"
        std::string age_group; // e.g. "25-34"
        float confidence;
        bool is_analyzed;      // True if attributes have been computed
    };

    struct CustomerResult {
        int track_id;
        postproc::BoundingBox box;
        CustomerAttributes attributes;
    };

    struct AnalyzerConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Model 1: Detector (Person Detection) ---
        std::string det_model_path; // e.g. yolov8_person.rknn
        int det_input_width = 640;
        int det_input_height = 640;
        float det_conf = 0.5f;

        // --- Model 2: Attribute Classifier ---
        std::string attr_model_path; // e.g. age_gender_net.xml
        std::string attr_labels_path;
        int attr_input_width = 224;
        int attr_input_height = 224;

        // --- Logic Settings ---
        // How often to re-scan attributes for a tracked person?
        // 0 = Every frame (slow), 30 = Once per second (at 30fps)
        int attribute_update_interval = 30;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class CustomerAnalyzer {
    public:
        explicit CustomerAnalyzer(const AnalyzerConfig& config);
        ~CustomerAnalyzer();

        // Move semantics
        CustomerAnalyzer(CustomerAnalyzer&&) noexcept;
        CustomerAnalyzer& operator=(CustomerAnalyzer&&) noexcept;
        CustomerAnalyzer(const CustomerAnalyzer&) = delete;
        CustomerAnalyzer& operator=(const CustomerAnalyzer&) = delete;

        /**
         * @brief Analyze a frame for customer insights.
         *
         * @param image Input CCTV frame.
         * @return List of tracked customers with attributes.
         */
        std::vector<CustomerResult> analyze(const cv::Mat& image);

        /**
         * @brief Reset tracking state (e.g., camera restart).
         */
        void reset();

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::retail