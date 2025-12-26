#pragma once

#include <string>
#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>

// Include Target enum
#include <xinfer/compiler/base_compiler.h>
#include <xinfer/postproc/vision/types.h> // For BoundingBox

namespace xinfer::zoo::vision {

    /**
     * @brief Result of Vehicle Identification.
     */
    struct VehicleResult {
        // Detection Info
        postproc::BoundingBox box;
        std::string type;       // "Car", "Bus", "Truck", "Motorcycle"

        // Attribute Info (if attribute model is enabled)
        std::string make_model; // e.g., "Toyota Camry"
        std::string color;      // e.g., "Silver"
        float attr_confidence;
    };

    struct VehicleIdentifierConfig {
        // Hardware Target
        xinfer::Target target = xinfer::Target::INTEL_OV;

        // --- Stage 1: Detector (YOLO) ---
        std::string det_model_path;
        int det_input_width = 640;
        int det_input_height = 640;
        float det_conf_thresh = 0.5f;
        float det_nms_thresh = 0.45f;
        std::string det_labels_path; // Maps class ID 0->Person, 2->Car, etc.

        // --- Stage 2: Attribute Classifier (Optional) ---
        // If empty, only detection is performed.
        std::string attr_model_path;
        std::string attr_labels_path; // Text file with Make/Model names
        int attr_input_width = 224;
        int attr_input_height = 224;

        // Vendor flags
        std::vector<std::string> vendor_params;
    };

    class VehicleIdentifier {
    public:
        explicit VehicleIdentifier(const VehicleIdentifierConfig& config);
        ~VehicleIdentifier();

        // Move semantics
        VehicleIdentifier(VehicleIdentifier&&) noexcept;
        VehicleIdentifier& operator=(VehicleIdentifier&&) noexcept;
        VehicleIdentifier(const VehicleIdentifier&) = delete;
        VehicleIdentifier& operator=(const VehicleIdentifier&) = delete;

        /**
         * @brief Identify vehicles in a scene.
         *
         * @param image Input frame.
         * @return List of vehicles with attributes.
         */
        std::vector<VehicleResult> identify(const cv::Mat& image);

    private:
        struct Impl;
        std::unique_ptr<Impl> pimpl_;
    };

} // namespace xinfer::zoo::vision